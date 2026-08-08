# CAP-EMF-1 extended audit and successor-trial requirements

**Date:** 2026-08-06
**Status:** post-run audit; no training or model artifacts modified
**Primary records:** `EncoderIndependentCAPEMF1Results.md`,
`EncoderIndependentCAPEMF1Retrospective.md`, and
`EncoderIndependentCAPEMF1Protocol.md`

## Executive judgment

CAP-EMF-1 is a legitimate, integrity-preserved **negative foundation
experiment**. It is considerably more informative than the earlier collapsed
raw-pixel attempts, but it is not yet a successful encoder-independent
drifting model.

The run established that a 37.7M-parameter raw-pixel network can learn
nontrivial CIFAR-10 structure and generate in exactly one model call without an
encoder, VAE, teacher, or perceptual loss. It did not produce a capable image
generator, and it did not test the later Anchored Self-Feature Drifting (ASFD)
correction. The correct project status is:

| Question | Verdict |
|---|---|
| Did the long run genuinely execute as reported? | Yes |
| Is inference exactly one model call? | Yes |
| Is training independent of an external encoder? | Yes |
| Did it avoid total collapse and literal copying? | Yes |
| Did it learn a good approximation of CIFAR-10? | No |
| Did it test encoder-independent drifting itself? | No--only its proposed foundation |
| Is the reported failure mechanism proven? | No |
| Is another run justified? | Yes, but only after checkpoint forensics and a controlled screen |
| Should the retrospective's proposed sampled-`r` floor remedy be launched as written? | No |

The results and retrospective are admirably candid, but several causal and
metric claims need correction before they guide another expensive run.

## 1. What the experiment actually accomplished

The durable result is narrower than some of the wording in the report, but it
is still useful.

The model completed 650,000 updates--41.6 million examples--with no non-finite
updates. The sealed checkpoint, source hashes, and principal evaluation
artifacts appear internally consistent. The uncurated samples contain
recognizable color arrangements, object fragments, and scene-like
organization. That is a genuine improvement over the project's earlier
raw-pixel models, which frequently collapsed or obtained essentially zero
recall.

The most meaningful distributional evidence is:

- repository-internal FID: **112.94**;
- precision: **0.372**;
- recall: **0.241**;
- coverage: **0.169**;
- effective-rank ratio: **8.44** times the real-data value;
- low-frequency Haar ratio: **0.931**;
- detail-band ratios: approximately **3.5--6.4** times the real-data values.

These statistics say that the generator learned broad color and spatial
organization, but its output distribution remained much too diffuse and
texture-heavy. "Object-like cues and layouts" is defensible. "Coherent scene
composition" is somewhat stronger than the grids consistently support.

The nearest-neighbor analysis rules out literal pixel duplication. It does not
prove complete absence of overfitting, nor does it prove that the low recall is
"genuine coverage." The safe claim is that the model was not copying exact
training images.

The exploratory 200k-window parameter average improving the internal FID from
112.9 to 83.7 is an important clue, but not a result that can currently be
reproduced: the relevant raw snapshots, reconstructed weights, hashes, and
complete reconstruction procedure were not preserved. It should be treated as
evidence for late optimization noise, not as a promoted model result.

## 2. How this fits the larger project

The formalization proves statements about ideal population drift fields: under
the verified conditions, an exactly vanishing field identifies the target
distribution. That does not imply that:

- a finite neural network can accurately represent the field;
- the stochastic objective trains it stably;
- the conditioning distribution adequately covers inference;
- raw-pixel errors align with semantic image quality;
- a one-step numerical approximation remains controlled.

CAP-EMF-1 investigates those approximation and optimization questions. It is
Stage A in `AnchoredSelfFeatureDriftingResearchPlan.md`, not the self-feature
drifting intervention itself.

Consequently, CAP-EMF-1 does not refute the identifiability theory, and it does
not show that encoder-independent drifting fails. It shows that the present
one-call raw-pixel foundation is not yet adequate enough to support the
drifting experiment.

There is also an important novelty boundary. Encoder-free, one-call generation
is already known to be possible in the wider literature. The result here is
valuable as an internal mechanism advance and diagnosis, not by itself as a new
general existence result.

## 3. Corrections to the results and retrospective

### 3.1 The reported FID is not directly comparable with DDPM/EDM numbers

The results document calls the value directly comparable with published
CIFAR-10 FIDs. The evaluator itself says otherwise. It uses torchvision's
ImageNet-classification Inception network, ImageNet normalization, and its own
resizing pipeline. Published FID numbers can change materially with feature
weights, resizing, quantization, and preprocessing. This is precisely the
problem addressed by CleanFID:

- Parmar et al., *On Aliased Resizing and Surprising Subtleties in GAN
  Evaluation*: <https://github.com/GaParmar/clean-fid>

Therefore:

- 112.94 is valid for comparisons made with this repository's same evaluator;
- it clearly indicates poor generation;
- the statement "35 times worse than DDPM" is not certified;
- the preserved checkpoint must be evaluated with CleanFID or an exact
  standard CIFAR-10 FID implementation before a successor is compared with it.

A known public CIFAR-10 checkpoint or sample set should also be passed through
both evaluators as a positive control. A real-versus-real split calibration is
needed to expose the metric's intrinsic floor and variance.

### 3.2 The displayed time table is one batch, not the run

The table used to diagnose endpoint starvation contains one logged batch of 64
rows at step 575,000. It is not a final-window or run-wide aggregate.

Aggregating all 1,300 logged batches gives 83,200 diagnostic rows:

| Local time `t` | Share | Raw MSE |
|---|---:|---:|
| 0--0.3 | 1.97% | 0.039 |
| 0.3--0.6 | 29.27% | 0.242 |
| 0.6--0.8 | 45.61% | 4.182 |
| 0.8--0.9 | 19.10% | 5.872 |
| 0.9--0.95 | 3.67% | 18.571 |
| 0.95--1 | 0.38% | 44.884 |

The endpoint neighborhood was sparse, not absent. Across 41.6 million
examples, 0.38% still corresponds to approximately 158,000 rows.

The stronger concern is **joint conditioning**. Inference evaluates the network
at

\[
  (t,h)=(1,1), \qquad h=t-r,
\]

so the underlying lower endpoint is `r = 0`. Under the present sampler, the
probability of reaching a small neighborhood such as `t > .95` and `h > .95`
is only on the order of \(10^{-5}\)--roughly 500 rows over the entire run. A
one-dimensional `t` histogram cannot expose that mismatch.

The next implementation must log the joint distribution of `t`, `r`, and
`h = t-r`, including diagonal versus active rows.

A read-only sensitivity sweep of the preserved 650k EMA checkpoint adds an
important refinement. Along the valid long-map boundary `r = 0`, hence `h = t`,
the relative RMS change from the exact inference output `(t,h) = (1,1)` was
approximately:

| Boundary point | Relative RMS change | Output cosine with `(1,1)` |
|---|---:|---:|
| `(.97,.97)` | 0.60 | 0.80 |
| `(.98,.98)` | 0.58 | 0.81 |
| `(.99,.99)` | 0.37 | 0.93 |
| `(.995,.995)` | 0.17 | 0.985 |

Holding `t = 1` and varying only `h` from `.98`, `.99`, and `.995` to `1`
changed the output by only about 2.6%, 1.9%, and 1.3%, respectively. This is a
continuity and sensitivity diagnostic, not a correctness target--the correct
map at nearby times need not be identical. Nevertheless, a 37% change over the
last 1% of absolute time is large enough that coarse `.95--1` bins are
inadequate.

Under CAP's logit-normal marginal, the entire 41.6M-example run is expected to
contain only approximately 2,300 rows above `t = .98`, about 40 above `.99`,
and fewer than one above `.995`. Thus the evidence now supports **both** parts
of the diagnosis:

- the full joint inference corner is poorly represented;
- absolute-`t` endpoint extrapolation is independently severe.

The `1000`-scaled sinusoidal time embedding is a plausible contributor to this
sensitivity, but changing its frequency scale would be a separate architectural
factor and must not be bundled into the sampler repair.

### 3.3 CAP regressed from an already-correct ordered-endpoint sampler

The Euler Mean Flow paper samples both endpoints independently from a base
distribution and orders them. Its default is uniform sampling; log-normal
sampling is reported for ImageNet. See Appendix C.1:

- Li et al., *Trajectory Consistency for One-Step Generation on Euler Mean
  Flows*: <https://arxiv.org/html/2602.02571>

In the repository's reversed clock, an ordered pair is

\[
  t=\max(U_1,U_2), \qquad r=\min(U_1,U_2).
\]

The earlier `stage_pmf/objective.py` already implemented the essential
paper-faithful construction: two independent draws, `max`/`min`, and the exact
diagonal mixture. CAP's `stage_cap/objective.py` replaced it with one
logit-normal draw followed by `r = t * Uniform`, and then clamped the sampled
`r` itself to 0.01. The protocol's statement that this was inherited unchanged
is therefore incorrect.

A five-million-sample numerical audit gives the following coefficient tails
for

\[
  c = (t-r-\delta)_+\,t/\max(r,0.02).
\]

| Sampler | q90 | q95 | q99 | q99.9 | `P(c > 7)` |
|---|---:|---:|---:|---:|---:|
| CAP: one draw plus `r=tU` | 2.59 | 5.76 | 21.07 | 36.85 | 4.15% |
| Ordered iid logit-normal | 0.451 | 0.693 | 1.398 | 2.836 | 0.0021% |

Thus CAP's rewrite made coefficients above seven approximately 2,000 times
more frequent than the repository's prior ordered-logit-normal sampler.

There is a genuine tradeoff:

- ordered logit-normal sampling controls the coefficient tail very well but
  gives poor coverage of the full long-interval inference corner;
- ordered uniform sampling gives much stronger full-triangle coverage and is
  the paper's default outside the ImageNet setting;
- CAP's conditional-uniform construction has both a heavy coefficient tail and
  inadequate joint endpoint coverage.

For the neighborhood `t > .95, h > .9`, Monte Carlo probabilities are
approximately:

| Sampler | Probability | Expected rows in 41.6M |
|---|---:|---:|
| CAP conditional-uniform | 0.0115% | about 4,800 |
| Ordered iid logit-normal | effectively negligible | effectively none |
| Ordered iid uniform | 0.374% | about 156,000 |

This evidence makes the sampler the leading concrete mechanism, but it argues
for a **three-arm screen**, not an improvised endpoint mixture.

### 3.4 Raising the sampled `r` floor is not a safe remedy

The retrospective suggests increasing the floor from 0.01 to 0.05. There are
two problems.

First, inference uses `r = 0`. Raising the sampled `r` floor moves training
farther from the inference condition.

Second, the stated coefficient reduction from approximately 35 to 7 is
numerically incorrect. Around `t = .85, r = .05`, the coefficient is still
approximately 13--14. Reaching approximately 7 would require a denominator
near 0.1.

The implementation must distinguish:

- the actual sampled and conditioned value of `r`;
- the numerical coefficient denominator floor;
- the time-loss weight floor;
- any state-update denominator.

The official construction clamps denominators for numerical safety. It does
not require removing the `r`-near-zero condition from training.

### 3.5 The adaptive-loss explanation is backwards as currently written

Let \(S=\lVert e\rVert^2\), and suppose the detached adaptive weight is

\[
  w=\frac{a}{S+\epsilon}.
\]

The optimized per-row loss is \(L=wS\), and because `w` is detached,

\[
  \nabla_e L=\frac{2a e}{S+\epsilon}.
\]

For a large residual,

\[
  \lVert\nabla_eL\rVert\approx\frac{2a}{\lVert e\rVert}.
\]

Enormous raw residuals are therefore downweighted in output-gradient magnitude.
They do not automatically dominate training merely because their raw MSE is
large.

They could still inject damaging parameter-gradient directions through the
network Jacobian, but that must be measured. The next run must record weighted
loss and gradient contribution by joint time stratum rather than infer
optimization importance from raw MSE or clipping frequency.

### 3.6 The convolutional refiner is not causing the excess detail

This is the most important saved-checkpoint finding.

The preserved `cap_emf1_step650000_ema.pt` checkpoint was loaded read-only on
CPU in evaluation mode. Using 512 deterministic endpoint noises, the
transformer/patch-head output was separated from the final refined output:

| Band variance | Before refiner | Final output | Final/base |
|---|---:|---:|---:|
| LL | 1.156 | 0.912 | 0.789 |
| LH | 0.398 | 0.112 | 0.282 |
| HL | 0.394 | 0.121 | 0.308 |
| HH | 0.266 | 0.0298 | **0.112** |

The refiner suppresses approximately 89% of the base output's HH energy. Its
residual has RMS approximately 0.48 times the base output and cosine similarity
\(-0.728\) with it. It is performing a large cancellation, not adding the
excess texture.

The 512-noise calculation is a source-attribution diagnostic, not a replacement
FID result. It should be repeated with the sealed 2,048 health latents and
stored as a formal artifact. The suppression is nevertheless so large that the
qualitative conclusion is robust.

This points upstream:

- the non-overlapping patch prediction head or trunk produces extreme
  high-frequency or patch-phase content;
- the refiner learns to cancel much of it;
- late optimization or EMA movement makes that cancellation fragile.

Removing the refiner would probably make the problem substantially worse. The
next run must expose and monitor:

- the pre-refiner output;
- the refiner residual;
- the final output;
- per-band energy and correlation for all three;
- phase-specific or checkerboard energy induced by the patch-2 decoder.

A post-hoc gain sweep of `base + alpha * refiner_residual` reinforces this
interpretation. Increasing the residual gain from `alpha = 1` to approximately
`1.25` reduced the estimated HH target ratio from about 6.36 to 4.24, after
which it began rising again. A modestly stronger cancellation helps, but does
not solve the upstream defect. Penalizing the refiner residual by itself would
be especially dangerous: it would force the final output back toward the
pathological high-frequency base.

An overlapping or anti-aliased output decoder may ultimately be appropriate,
but it should be a separate architecture experiment after the sampler screen.
If an auxiliary multiscale constraint is introduced, it should supervise the
base head as well as the final output rather than asking the residual refiner to
hide an unconstrained base.

### 3.7 The production finite difference is outside the audited local regime

This moved from a hypothetical precision concern to a directly observed
mechanism during the post-run audit.

CAP uses `delta = .01`, but `ScalarEmbedding` constructs angles as

\[
  1000\,t\,\omega_j,
\]

where the largest frequency is \(\omega_j=1\). A single EMF finite-difference
step therefore moves the highest-frequency time feature by **10 radians**. Its
period in normalized time is only \(2\pi/1000\approx .00628\), which is smaller
than the production step. In the actual conditioning geometry, `.01` is not a
small perturbation.

A read-only exact-JVP audit on the preserved trained 650k EMA checkpoint found:

| Stratum | `delta=.01` relative RMS error | Quotient/JVP cosine |
|---|---:|---:|
| `t=.98, r=.02`, four rows | 0.913 | 0.501 |
| `t=.85, r=.10`, four rows | 0.525 | 0.817 |

Separate single-row checks gave the same qualitative result. At `delta=.0002`,
the corresponding errors fell to approximately 0.036 and 0.112, with cosines
0.9996 and 0.9939. This confirms that the implementation approaches the correct
direction, while showing that the **production** step can point substantially
away from the local derivative. Non-monotone behavior at `.005` and `.0025` is
consistent with the sinusoidal phase being crossed rather than with a clean
local-error regime.

This does not prove that finite-difference mismatch alone caused the final FID:
the trained network has adapted to the stopped-difference objective, and EMF's
surrogate is not required to be an exact JVP. It does, however, invalidate the
claim that the production surrogate was already numerically certified. The
method's local-linear justification becomes weak when the measured direction
has cosine about 0.5 with the derivative it is meant to approximate.

The current preflight uses a small artificially activated model on CPU in
float64 and tests deltas `1e-3`, `1e-4`, and `1e-5`, not the production `.01`.
It verifies convergence to the algebraically correct limit; it explicitly does
not certify the accuracy of the chosen production step.

TF32 adds a separate concern. Values produced by TF32 matrix multiplication
have already incurred reduced-mantissa rounding; subtracting them in FP32 does
not restore lost information, and division by a small delta can amplify
mismatched errors.

There is a second production-only risk. The active-row optimization computes
the current prediction in a full batch of 64, but computes future predictions
in a variable-size gathered sub-batch. GPU kernel and batch-shape rounding
differences therefore enter a subtraction that is divided by `.01`. The
float64 sparse-versus-dense test cannot rule this out.

Before another run, the trained checkpoint must be tested at representative
`(t,r)` and active-count strata using:

- TF32 enabled versus disabled;
- `delta` in `{.02, .01, .005, .0025}`;
- finite difference versus exact JVP where feasible;
- sparse active-row evaluation versus fixed-shape dense evaluation;
- target and gradient RMS, relative error, cosine similarity, and spatial
  spectrum.

If sparse execution is unsafe, stopped future paths should be evaluated densely
or padded to a fixed shape. If TF32 is unsafe only for the quotient path, that
path can be forced to full FP32 without necessarily abandoning TF32 for the
entire trunk.

Before training, a joint sweep over time-embedding frequency scale, delta,
precision, and dense-versus-sparse evaluation must select a configuration that
passes a predeclared quotient/JVP target-and-gradient cosine threshold across
the actual sampler strata. Simply lowering delta under TF32 is not safe:
subtractive roundoff can eventually dominate. The smallest stable delta under
FP32 or a smoother/lower-frequency time embedding are the two principled repair
families, and they must be compared rather than guessed.

The official EMF work successfully uses mixed precision, so TF32 itself should
not be declared the cause. The directly established issue is the interaction
between CAP's `1000`-scaled time embedding and its unaudited production delta.

### 3.8 Late divergence and clipping statements need qualification

The final preserved EMA checkpoint clearly has excessive high-frequency
energy. But the persisted training health stream evaluated raw online weights,
whereas the table in the results document is labelled EMA. The intermediate EMA
checkpoints were not preserved, so "monotone EMA divergence" cannot now be
fully audited.

The recorded H7 final-window clipping pass is also invalid because recovery
inserted zero when the actual final-window counter was unavailable. However,
the 15.3% figure is run-wide, not the final 20,000-update rate, and cannot be
substituted for H7. Sparse logs suggest clipping declined substantially late in
training.

The conclusion "the final model has severe excess detail" is solid. The
proposed clipping-driven causal narrative is not.

### 3.9 The training trajectory and post-hoc average were not preserved well

The report's intermediate EMA trajectory cannot be reconstructed from the
remaining artifacts. The claimed 200k post-hoc average is likewise not
reconstructible and may be closer to snapshot weight averaging than to a
precisely defined exponential moving average.

Post-hoc EMA is a legitimate technique, but it requires raw snapshots and an
exact reconstruction recipe. NVIDIA's EDM2 implementation is a useful
reference for preserving and reconstructing EMA profiles:

- <https://github.com/NVlabs/edm2>

The next trial should either maintain several predeclared EMA horizons or save
enough raw snapshots to synthesize them exactly after training. Every reported
secondary result must have reconstructible weights and hashes.

## 4. Ranked explanation of the failure

### 4.1 Production delta versus time-conditioning frequency -- high, directly evidenced

At the preserved checkpoint, the `.01` quotient can have only approximately
0.5 cosine with the exact directional derivative. CAP's preflight certified the
limit and sign, but not the production setting. This is co-primary with the
sampler regression and must be repaired before the sampler arms are meaningful.

### 4.2 Custom sampler and joint inference-condition mismatch -- high

CAP replaced an already-correct ordered-endpoint sampler with a construction
that creates a much heavier coefficient tail while still poorly covering the
one-step inference corner. This is the strongest code- and literature-backed
mechanism.

### 4.3 Fragile patch-head/refiner cancellation -- high

The base decoder produces extreme detail and relies on the refiner to cancel
it. This directly explains how broad structure can improve while fine
structure remains excessive or deteriorates late.

### 4.4 Late optimizer/EMA noise -- moderate to high

The constant learning rate, small batch, late degradation, and apparent benefit
from much longer weight averaging all point toward oscillation. The loss of the
post-hoc weights prevents a definitive conclusion.

### 4.5 TF32 and sparse-gather amplification -- plausible, untested

These production execution effects are distinct from the now-demonstrated
delta/embedding mismatch. They could further destabilize the finite-difference
target and fragile high-frequency cancellation. A trained-checkpoint GPU audit
can settle them without another full run.

### 4.6 Raw-pixel objective burden -- structural

Pixel losses spend capacity on details that are not necessarily perceptually
important. Recent work such as PixelGen addresses this with learned perceptual
supervision:

- *PixelGen: Improving Pixel Diffusion with Perceptual Supervision*:
  <https://arxiv.org/html/2602.02493>

That exact remedy would violate this project's strict encoder-free-training
objective. Analytic multiscale safeguards or features derived from the
generator itself remain compatible possibilities after the foundation is
stable.

### 4.7 Insufficient total compute -- low as a first response

The run already processed 41.6 million examples, a scale comparable to the
official EMF pixel experiment's training exposure, albeit on a different
dataset and architecture. More importantly, CAP-EMF-1 worsened late. Simply
extending it to 750k or beyond is not justified.

## 5. Mandatory work before launching another training trial

### 5.1 Phase 0A: measurement integrity

1. Re-evaluate the 650k checkpoint with standard CleanFID while retaining the
   existing metric as a legacy within-repository series.
2. Run real-versus-real calibration and a known public CIFAR-10 model or sample
   set through both metric pipelines.
3. Report raw and clipped outputs separately for Haar energy, spectrum, rank,
   and saturation.
4. Use evaluation-only semantic metrics or a classifier recognizability check.
   External encoders are prohibited for training, not for honest evaluation.
5. Treat the already-opened CIFAR-10 test set as report-only, not newly sealed,
   during the next development cycle.

### 5.2 Phase 0B: saved-checkpoint mechanism audit

1. Repeat the pre-refiner/refiner/final decomposition on the sealed health
   latents and save it as an artifact.
2. Measure patch-phase/checkerboard energy and the spectrum of the refiner
   residual.
3. Sweep the checkpoint's endpoint response over a two-dimensional `(t,h)`
   grid, with special focus on the inference corner `(1,1)`.
4. Treat the CPU checkpoint JVP result as a failed production-delta audit and
   reproduce it on the target GPU. Jointly sweep embedding scale, delta,
   TF32-versus-FP32, and sparse-versus-dense evaluation; compare both targets
   and parameter gradients with an exact JVP reference on representative
   strata.
5. Select and freeze a numerically justified local-difference configuration
   before starting any sampler training comparison. Do not carry `.01` forward
   merely because it was used historically.
6. Simulate and unit-test every candidate sampler against analytic joint-bin
   probabilities before training.

### 5.3 Phase 0C: logging, gate, and artifact repair

1. Replace one-batch diagnostic tables with running sufficient statistics over
   every example.
2. Save joint `t/r/h` bins, diagonal/active status, coefficient and target
   quantiles, adaptive weight, weighted loss, output-gradient contribution,
   and sampled parameter-gradient contribution.
3. Evaluate both raw and EMA weights at every checkpoint.
4. Log base-head, refiner-residual, and final-output diagnostics separately.
5. Preserve actual total and final-window clipping counters across recovery.
6. Save and continuously sync every required raw checkpoint, EMA checkpoint,
   optimizer state, source manifest, and reconstruction recipe.
7. Include every executable entry point, including the unit runner, in the
   source manifest.
8. Record runtime overrides such as the actual 64-by-1 microbatch split in the
   final artifact rather than only in a transient log.
9. Make wrappers return nonzero on a Python traceback and test an actual GPU
   resume before launch.
10. Benchmark hundreds of complete training-loop updates, including data
    loading, transfers, EMA, diagnostics, checkpointing, and upload overhead.
11. Replace one-sided capability floors with real-data-calibrated lower and
    upper bounds. Rank and every Haar band must be able to fail by being either
    too small or too large.
12. Rewrite the frozen protocol so it no longer mixes the final all-class 650k
    configuration with stale 160k automobile language.

## 6. Recommended successor experiment

### 6.1 Numerical admission gate

The sampler comparison is not interpretable until the local surrogate is in a
credible numerical regime. Before the three training arms, use the frozen
checkpoint and a small short-training screen to choose between:

- the existing embedding scale with a substantially smaller, fully FP32,
  fixed-shape stopped difference;
- a smoother or lower-frequency scalar embedding with a delta large enough to
  avoid cancellation but small enough to remain local.

Predeclare an acceptable quotient-versus-exact-JVP cosine and relative-error
range across the sampler's common `(t,r)` strata, including the endpoint and
large-coefficient tails. Confirm gradient direction as well as target
direction. A setting is admissible only if it passes on the production GPU.

This screen changes the common numerical foundation for all subsequent sampler
arms. It is a prerequisite, not another factor to vary inside the sampler
comparison.

### 6.2 Scientific question

The clean next question is:

> Which ordered-endpoint distribution controls the EMF coefficient while still
> providing enough coverage of the one-step inference corner?

This is narrower and better supported than a bundled "endpoint fraction plus
larger sampled-`r` floor" intervention.

### 6.3 Three matched sampler arms

Use identical initialization, data order, base noise, augmentation stream,
architecture, optimizer, LR, adaptive loss, the newly audited common
finite-difference setting, refiner, and EMA.

| Arm | Sampler | Purpose |
|---|---|---|
| A: historical control | CAP logit-normal `t`, then `r=tU` | Reproduce the current mechanism |
| B: ordered-logit-normal | Two iid CAP logit-normal draws, then max/min | Isolate the sampler regression and coefficient tail |
| C: ordered-uniform | Two iid uniform draws, then max/min | Test the paper default and inference-corner coverage |

Keep the same declared diagonal fraction in every arm. In B and C, condition on
the actual ordered `r`, including values near zero, while clamping only the
coefficient denominator at 0.02.

Arm B is the closest repair to CAP's declared marginal time law, but it is not
a literal one-factor arm: like C, it replaces the conditional pair and removes
the sampled-`r` floor. Arm C is the strongest paper-backed candidate for the
full inference-support problem. The screen therefore estimates the effect of
each repaired package; it does not separately identify ordering and floor
removal.

Do not simultaneously:

- raise the sampled `r` floor;
- change the refiner or patch decoder;
- add an ad hoc endpoint loss;
- add ASFD;
- alter the LR schedule;
- change the EMA selection rule.

A naive endpoint MSE pairing independent noise directly with an arbitrary clean
image is unsafe: the deterministic MSE optimum can be the conditional mean and
encourage collapse. Any explicit endpoint mixture must use the correctly
derived EMF target.

### 6.4 Staged execution

Checkpoints and complete artifacts should be retained at least every 50k
updates.

- **50k:** stop every arm; rerun the complete numerical matrix on its exact raw
  checkpoint and require the immutable continuation gate.
- **150k:** early capability and gradient-distribution check.
- **300k:** first scientifically meaningful comparison. The previous run's
  harmful separation appeared later than the earliest checkpoints.
- **Beyond 300k:** continue only arms whose final and upstream high-frequency
  diagnostics remain stable.

If budget permits, use two matched seeds. One three-arm seed remains
developmental. The implemented successive-halving rule takes all three arms
to 150k, retains the concurrent legacy control, and promotes exactly one
ordered arm only if it beats that control beyond the predeclared standard-
metric margins. A single historical-control comparison is cheaper but gives
weaker causal evidence.

### 6.5 Predeclared development verdict

An arm is promotable only if it jointly satisfies:

- CleanFID and CleanKID better than the re-evaluated CAP-EMF-1 checkpoint by
  more than the recorded real/real discrepancy, plus lower shared-backend
  unbiased repository KID at the fixed comparison step;
- no loss of both precision and recall;
- effective rank inside a calibrated two-sided real-data interval;
- every Haar band inside calibrated lower and upper intervals;
- no late increase in base-head or final HH energy;
- stable inference-corner error: in each predeclared late 25k window, at least
  1,024 all-row observations with `t > .95` and `h > .90`, and no more than a
  four-fold increase in mean raw MSE from `(100k,125k]` to `(125k,150k]`;
- bounded raw-output saturation;
- acceptable genuine final-window clipping and no non-finite updates;
- an intact, uncurated fixed-seed grid for descriptive review; visibly broken
  output may veto continuation, but subjective appearance cannot create GO;
- exactly one inference call.

The test split should not be used to select among development arms.

## 7. What to do after the sampler screen

Choose exactly one next factor from the evidence:

- If the base output remains extreme under all samplers, test an overlapping or
  anti-aliased output decoder, or an explicitly band-controlled synthesis head.
- If the refiner cancellation is initially healthy but drifts late, test an LR
  decay schedule or a better-conditioned/gated residual correction.
- If full FP32 corrects target or gradient disagreement, isolate the
  finite-difference path in full FP32.
- If sparse active-row execution disagrees with dense execution, use a fixed
  batch shape for the stopped future evaluation.
- If late oscillation remains, test multiple predeclared EMA horizons or exact
  reconstructible post-hoc EMA from synchronized raw snapshots.

Do not lead with a refiner-off experiment. The preserved checkpoint strongly
predicts that removing it will expose substantially more high-frequency noise.

Only after one sampler and one independently supported stability or decoder
repair have passed development gates should another 600k-scale capability run
be authorized. A replication is required before a general claim.

## 8. ASFD decision

ASFD training should remain blocked. Its own roadmap requires a capable
one-call foundation and explicitly stops when that prerequisite is absent.

The present output-space failure does not mathematically prove that every
internal representation is useless. A read-only feature-qualification audit is
reasonable. An expensive ASFD training fork is not: adding a drift correction
to a generator whose base output and temporal training geometry are unstable
would confound the diagnosis.

## 9. Final recommendation

Do not launch another 600k--750k run using the retrospective's combined
boundary-mixture and raised sampled-`r` floor recipe.

First complete the saved-checkpoint numerical and decoder forensics. The
`.01`/1000-scale mismatch must be discharged by choosing an audited numerical
configuration before any further training result can be interpreted cleanly.
Then run the matched three-arm sampler screen with repaired logging and artifact
preservation. If ordered-uniform improves endpoint behavior without recreating
the coefficient pathology, it becomes the leading foundation configuration. If
ordered-logit-normal stabilizes training but misses the inference corner, that
clarifies that a principled mixture or reweighted ordered distribution--not a
sampled-`r` floor--is the next mathematical design problem. If neither helps,
the output decoder should be repaired before more large-scale training.

The experiment was not wasted. It moved the project from "raw one-step
training collapses" to a sharper, actionable diagnosis:

1. the network learns coarse distributional structure;
2. the production `.01` finite difference is not local relative to the
   `1000`-scaled time embedding;
3. CAP regressed from a safer ordered-endpoint sampler;
4. the actual inference corner is poorly represented;
5. the patch head creates extreme fine-scale content;
6. the refiner performs a fragile cancellation;
7. late weight stability and production TF32/sparse execution remain
   unresolved.

That is enough evidence to justify a carefully controlled successor screen,
but not another blind long run.

## 10. Implementation status (2026-08-06)

The successor infrastructure described above is implemented in
`numerics/encoder_independent_drifting/stage_cap2/`, with its frozen execution
order in `numerics/EncoderIndependentCAPEMF2ScreenProtocol.md`.

Completed locally:

- explicit embedding-scale, delta, sampler, sampled-`r`, denominator, and
  stopped-execution controls;
- matched dense and dense-FP32 stopped paths;
- ordered logit-normal and ordered-uniform samplers alongside the historical
  control;
- all-row joint objective ledgers;
- raw/EMA base-head, refiner-residual, and final-output health;
- two-sided calibrated-gate machinery and honest recovery counters;
- trained-checkpoint JVP/gradient admission tooling;
- full-loop benchmark and source-bound preflight tooling;
- cadence-adjusted benchmark accounting that measures ordinary 512-sample and
  checkpoint 2,048-sample health paths separately from loop and artifact I/O;
- guarded 50k/100k/150k/300k screen runner with a mandatory 50k raw-state
  numerical stop before continuation;
- CleanFID PNG export and positive-control evaluation path;
- disjoint train/train CleanFID finite-sample discrepancy calibration tooling;
- fixed-latent checkpoint forensics covering raw/clipped base, residual, and
  final outputs, period-two phase energy, spectra, and a `(t,h)` grid;
- exact binding of numerical admission, benchmark, and training to one
  declared GPU model;
- explicit source manifests and artifact hashes;
- a pinned NVIDIA StyleGAN2-ADA positive-control generator with exact upstream
  commit, checkpoint SHA, balanced seed/class rule, immutable PNG provenance,
  and no route into training;
- a fail-closed production-readiness launcher that can run only numerical
  admission, same-environment forensics, the measured benchmark, and preflight.
- independent fixed-count diagonal RNG with the intended unsorted-first-draw
  marginal, eliminating the diagonal `max` bias found in the audit;
- 63-batch homogeneous numerical admission over real CIFAR-10 and two synthetic
  stress sources, seven time strata (including a below-floor row), quotient,
  assembled target, and global
  parameter-gradient fidelity, plus heterogeneous 4-row and exact production-
  shape 16-row diagonal/active gradient batches;
- actual trainable-parameter accounting separated from serialized buffers;
- atomic, hash-bound JSON/checkpoint/snapshot/recovery artifacts, an immutable
  50k raw-state continuation certificate, raw-state 150k re-admission, and a
  three-arm selection certificate layered over individual 150k eligibility;
- one shared CAP1/CAP2 50k CleanFID/CleanKID evaluator, fixed auxiliary KID and
  precision/recall, literal-pixel memorization audit, and mandatory sourced
  positive control;
- fail-closed preflight schema validation for every required sample count,
  seed, matrix stratum, hardware identity, metric version, and artifact source.
- a fixed all-row late inference-corner non-explosion gate and re-derived
  numerical/gate-calibration decisions at every authorization boundary.

The audited two-million-draw sampler artifact was regenerated from the final
source boundary and returned `GO`. It exposes the intended scientific
tradeoff:

| Arm | q99 coefficient | `P(c>7)` | `P(t>.95,h>.90)` |
|---|---:|---:|---:|
| historical CAP | 21.39 | 4.2219% | 0.00985% |
| ordered logit-normal | 1.43 | 0.00245% | 0% observed |
| ordered uniform | 22.36 | 4.0523% | 0.37055% |

The gate calibration was likewise regenerated and returned `GO`: 12 globally
disjoint pairs of 2,048 training images consume 49,152 of the 50,000 training
rows without overlap. The derived two-sided gate now independently covers
moment, variance, effective rank, and LL/LH/HL/HH energy. Its serialized gate
is reconstructed from the stored observations whenever preflight is loaded.

The current local verification covers 121 CAP/CAP2 tests spanning numerical,
sampler, development-evaluation, recovery, artifact, promotion, early-
admission, selection, benchmark-cadence, and preflight behavior. All passed,
as did lint, byte-compilation, CAP2 CLI entry-point checks, and the complete
older encoder-independent regression harness. All three CAP2 arms complete
the enhanced CPU smoke-training path.

The local pre-budget metric evidence is now complete and source-current:

- a hash-sealed full-train CleanFID feature population supplies the common KID
  reference (`621febf9...62706db`);
- the disjoint real/train-halves point is CleanFID `2.1363`, CleanKID
  `0.0000156`;
- the preserved CAP-EMF-1 EMA is CleanFID `128.8825`, CleanKID `0.113611`,
  precision `0.3833`, recall `0.3149`;
- the pinned NVIDIA StyleGAN2-ADA positive control is CleanFID `3.1933`,
  CleanKID `0.000661`, precision `0.7144`, recall `0.6836`;
- baseline, control, and calibration share the exact reference hash and the
  same recorded Python/PyTorch/torchvision/CleanFID environment;
- the final sampler audit and two-sided gate calibration both return `GO` from
  the same live source manifest.

Still intentionally blocked, and necessarily production-specific:

1. the 2,048-latent saved-checkpoint mechanism-forensics artifact on the
   rented training GPU;
2. full numerical admission on the preserved checkpoint on that same GPU;
3. the 2,000-step exact-loop benchmark and measured price projection there;
4. source-bound fail-closed CAP2 preflight assembled from those results;
5. any 50k or longer training arm.

No paid or long training run has been launched. The single
`stage_cap2.production_readiness` entry point performs only the four remaining
gates and has no route to training. All must return `GO` before a screen starts.
