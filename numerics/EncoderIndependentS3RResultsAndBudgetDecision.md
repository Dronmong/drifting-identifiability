# S3R result diagnosis and decision to enter the budget run

**Date:** 2026-08-03  
**Status:** one-unit developmental screen complete; S3R is closed  
**Decision:** do not rerun or retune S3R locally; build the cloud-scale
`CAP-EMF-1` capability experiment next

## 1. Executive conclusion

The strict result is unchanged: **none of the three S3R arms passed the
predeclared train-only gate**. This cannot be relabeled as a pass after seeing
the data.

The run nevertheless made a useful mechanism choice:

- continuous pMF reproduced a genuine low-rank collapse;
- AlphaFlow prevented that collapse but remained substantially
  under-dispersed and weak in image-detail bands; and
- direct-`x` Euler Mean Flow (EMF) was the strongest arm by a clear margin. It
  nearly restored raw amplitude, retained broad rank, had the healthiest raw
  multiscale spectrum, and took about half the time of continuous pMF.

EMF is therefore promoted only as the **engineering choice for the next
capability experiment**. It is not promoted as a successful image generator,
because S3R intentionally did not inspect the official test set, compute image
quality metrics, or produce a sealed visual confirmation.

The most likely remaining obstruction is no longer primarily the continuous
MeanFlow objective. It is the combination of:

1. a coarse 8-by-8 spatial-token grid;
2. only 12,500 optimizer updates;
3. a fixed EMA that is immature at that horizon; and
4. a small-sample health gate with one directionally incorrect retention
   check.

The correct response is **not** another local S3R repair loop. The next
training experiment should be the rented-GPU run, with a finer and more
faithful direct-pixel architecture, a materially longer EMF training horizon,
and corrected diagnostics frozen before launch.

## 2. What was actually run

All three arms used the same:

- CIFAR-10 automobile training pool: 5,000 images;
- raw 32-by-32 RGB pixels, with no VAE or learned feature encoder;
- one-call endpoint sampler;
- width-384, depth-12, eight-head U-shaped transformer;
- patch size 4, hence only 64 image tokens;
- effective batch 64;
- AdamW at `1e-4`, no weight decay;
- 12,500 updates / 800,000 examples / 160 nominal class epochs;
- fixed train-only endpoint noise and target samples; and
- `EMA = 0.9999` and gradient clipping at norm 10.

The arms isolated three objective mechanisms:

| arm | mechanism | JVP during training? |
|---|---|---:|
| `pmf` | continuous pixel MeanFlow with a four-block unshared auxiliary branch | yes |
| `alpha` | discrete AlphaFlow curriculum, floored at `alpha=0.005` | no |
| `emf` | direct-`x` Euler Mean Flow with a stopped local finite difference | no |

The endpoint sampler for every arm is exactly one model invocation. The
experiment therefore tested a real one-call, raw-pixel foundation, but **not a
drifting correction**. Laplace/Sinkhorn terms were deliberately absent so a
failed generator foundation could not be mistaken for a failed drifting
field.

## 3. Artifact integrity and claim boundary

The four result hashes verify:

| artifact | SHA-256 valid? |
|---|---:|
| `s3r_alpha_unit_800.json` | yes |
| `s3r_emf_unit_800.json` | yes |
| `s3r_pmf_unit_800.json` | yes |
| `s3r_unit_800_summary.json` | yes |

Each arm reports exactly 800,000 examples and the same source digest
`13c7066c...a2d7b9`. No official test image or learned evaluation feature was
used to choose the arm.

This gives a valid **developmental mechanism screen**. One unit cannot prove
replicability, image quality, or superiority. The current conclusion is about
which objective is least unhealthy under the fixed local setup.

## 4. Final result

The fixed target cloud had effective rank 8.89. Ratios above one mean that the
generated cloud used more measured directions than this small target cloud;
they do not by themselves mean better images.

| arm | raw moment | raw variance | raw rank ratio | EMA moment | EMA variance | EMA rank ratio | clip fraction | time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AlphaFlow | 0.425 | 0.424 | 1.240 | 0.218 | 0.199 | 3.881 | 0.000% | 1.09 h |
| **EMF** | **0.495** | **0.504** | **1.661** | **0.424** | **0.449** | **5.119** | 4.992% | 1.34 h |
| pMF | 0.512 | 0.439 | **0.349** | 0.223 | 0.155 | 2.723 | 0.000% | 2.51 h |

No arm passed:

- AlphaFlow failed the amplitude requirement.
- EMF failed the amplitude requirement as written and the historical
  rank-retention rule.
- pMF failed amplitude, rank, and rank retention.

## 5. What happened inside each arm

### 5.1 Continuous pMF: the original failure survived the repair

pMF reached raw second moment 0.512, but its centered variance was only 0.439
and its effective-rank ratio fell to 0.349, or an absolute rank of about 3.10.
This is the same qualitative defect as the earlier S3 run, whose final EMA
rank was about 2.5.

The objective did improve numerically, but remained much harder than the
others:

- mean raw error: `6.000 -> 1.431` from the first to last ten logs;
- interior error: `11.632 -> 2.667`;
- maximum logged JVP p90: `31.739`;
- six TFM/trajectory-consistency probes included three negative gradient
  cosines; and
- runtime was 2.51 hours, almost twice EMF.

The multiscale endpoint makes the collapse visible. Final raw variance ratios
were:

| LL | LH | HL | HH |
|---:|---:|---:|---:|
| 0.471 | 0.051 | 0.100 | 0.020 |

The model learned coarse low-frequency mass while losing nearly all diagonal
high-frequency variation. A deeper auxiliary branch did not remove the
self-referential JVP instability or the endpoint collapse. Continuous pMF
should therefore be retired from the immediate budget path.

### 5.2 AlphaFlow: stable, but the short curriculum underfit the endpoint

AlphaFlow used no JVP, never clipped, and retained rank. Its raw error fell
from 0.334 to 0.096. This establishes that the discrete curriculum removed a
large part of the instability seen in continuous pMF.

It did not learn enough endpoint amplitude: raw moment/variance were both
about 0.425 and EMA values were near 0.2. Its final raw high-high variance was
only 0.070.

The reported interior error approaching zero must not be read as a solved
interior map. At the end of the curriculum, the interior residual is scaled by
the frozen `alpha=0.005` target and adaptive numerator. The small number partly
reflects the objective construction. The endpoint-health metrics are the more
honest evidence, and they show stable underfitting.

This local schedule compressed AlphaFlow's literature-scale gradual transition
into roughly 10,000 updates. The AlphaFlow paper reports that a longer,
smoother transition improves quality and studies 400,000-step and longer
training regimes. AlphaFlow remains mathematically plausible, but the present
run gives no reason to spend the one budget run comparing it again with EMF.

### 5.3 EMF: a real optimization repair, but not yet a quality result

EMF delivered the strongest endpoint:

- raw moment `0.495` and centered variance `0.504`;
- raw rank ratio `1.661`, with no low-rank collapse;
- diagonal error `0.157 -> 0.091`;
- interior error `0.424 -> 0.156`;
- 1.34-hour runtime and only about 1.40 GB peak allocated memory; and
- no JVP.

Its final raw Haar variance ratios were:

| LL | LH | HL | HH |
|---:|---:|---:|---:|
| 0.509 | 0.410 | 0.512 | 0.159 |

This is much more balanced than pMF or AlphaFlow. It is the only arm that
simultaneously approached target bulk variance, retained broad diversity, and
carried substantial horizontal/vertical detail.

Two warnings remain:

1. high-high detail is still only 16% of the target; and
2. clipping occurred on 4.992% of updates, essentially on the predeclared 5%
   boundary, with maximum pre-clip norm 24.34.

EMF therefore repaired the **optimization mechanism**, not yet the image
generation problem. Its profile is exactly what should be tested with more
spatial resolution and a real training horizon.

## 6. The gate audit: real failures versus protocol artifacts

The original `FAIL` result must remain recorded. The following issues are
repairs for future protocols, not permission to rewrite S3R.

### 6.1 The EMA was not mature

With decay 0.9999, the EMA half-life is

\[
\frac{\log(1/2)}{\log(0.9999)}=6931\ \text{updates}.
\]

At update 12,500, an EMA initialized from the starting network still assigns

\[
0.9999^{12500}=0.2865
\]

of its total weight to that initialization. Requiring both raw and EMA
amplitude to exceed 0.5 was therefore poorly matched to this short screen.
This explains much of the EMA amplitude deficit, but it does not repair the
raw high-frequency deficit or establish recognizable images.

Future runs should record EMA length in updates and images, retain raw
checkpoints, and either use a warm-up/ramped EMA or judge the EMA only after a
predeclared number of half-lives. Post-hoc EMA reconstruction is also a
literature-supported option if the required parameter snapshots are stored.

### 6.2 Rank retention penalized improvement toward the target

The gate used

\[
\frac{\text{final rank ratio}}{\max_t\text{rank ratio}_t}\ge 0.8.
\]

EMF began at rank ratio 4.056 while its amplitude was only 0.34, then ended at
1.661 with target ratio 1. The gate called this `0.410` retention and failed
it. That is directionally wrong: moving from an over-dispersed, potentially
noise-like rank toward the target is not evidence of collapse.

A corrected future rule is

\[
R_{\rm final}\ge 0.8\min(R_{\rm best},1),
\]

combined with amplitude and multiscale energy requirements. This still rejects
pMF (`0.349 < 0.8`) and accepts rank as noncollapsed for AlphaFlow and EMF.

### 6.3 The endpoint cloud was too small for a sharp spectral decision

Only 64 fixed samples determined the rank and Haar snapshots. That is useful
for early collapse detection but too noisy for a threshold-level final
decision. The cloud also makes rank ratios above one hard to interpret. The
budget run should use at least 512 fixed train-only endpoints for cheap checks
and 2,048 for checkpoint audits.

### 6.4 EMF's raw moment miss was numerically tiny

The raw second-moment ratio was 0.49494 versus a 0.5 threshold, while centered
variance was 0.50436. The deterministic gate correctly failed it; scientifically,
this difference is not robust evidence of a broken generator. The meaningful
unresolved defect is the high-frequency imbalance and unknown recognizability,
not the 0.005 threshold miss.

## 7. Cross-check against the literature

The local ordering is consistent with the primary papers:

- AlphaFlow identifies optimization conflict between trajectory flow matching
  and consistency and improves it through a gradual curriculum. Its published
  studies use hundreds of thousands to more than a million updates, batch 256,
  and much larger DiTs.
- EMF removes the JVP and reports more stable training, about 50% lower
  training time/memory than MeanFlow in its image comparisons, and better FID
  under matched architectures. Its unconditional pixel experiment uses
  direct-`x` JiT-B/16, batch 64, 600,000 updates, Adam at `1e-4`, and EMA
  0.9999.
- The pMF paper shows that direct clean-image prediction is valuable, but also
  reports a large optimizer effect and a major gain from learned perceptual
  losses. Those learned losses are intentionally excluded here because the
  experiment is testing encoder independence.
- JiT supports direct clean-image prediction in raw pixels with no tokenizer,
  pretraining, or auxiliary learned representation.
- PixelDiT independently argues that local pixel-level modeling is important
  for pixel-space generation. That supports adding spatial/local capacity, but
  does not justify importing its entire dual-level system into a last-budget
  run without a separate port audit.

There is also a large scale mismatch. S3R used 12,500 updates. EMF's published
pixel result uses 600,000. pMF's 160-epoch ImageNet ablation is about 200,000
optimizer updates because the dataset and batch are much larger. Matching only
nominal epochs on a 5,000-image class does not reproduce the number of
optimization steps or the diversity of training examples.

## 8. What we learned

### Robust lessons

1. **The continuous JVP objective is not the best use of compute here.** It is
   slower, less stable, and collapses rank even after the auxiliary repair.
2. **Direct endpoint supervision matters.** EMF was the only arm to recover
   nearly correct raw variance without collapse.
3. **Objective stability is necessary but insufficient.** AlphaFlow is stable
   yet underfits; EMF is stable yet still lacks high-high detail.
4. **The present architecture is spatially coarse.** Patch 4 at 32-by-32 gives
   only 64 tokens. EMF's 256-by-256 JiT-B/16 uses 256 tokens. A 32-by-32
   analogue should use patch 2.
5. **A short train-only health screen cannot substitute for a capability
   run.** It can reject collapse but cannot prove recognizability.
6. **Drifting must remain downstream of a working one-step foundation.** A
   correction cannot manufacture spatial semantics that the endpoint model
   never learned.

### Things this run did not show

- It did not show that EMF generates recognizable automobiles.
- It did not show that EMF beats the original drifting paper.
- It did not test a Laplace or Sinkhorn correction.
- It did not show that learned encoders are unnecessary at ImageNet scale.
- It did not justify a second local seed of the same architecture.

## 9. Decision on the experiment queue

### Close S3R now

Do not run a second S3R unit and do not tune its thresholds. The one-unit screen
already answered its intended mechanism question. More local repetitions would
measure uncertainty around a setup that is too spatially coarse and too short
to answer the capability question.

### Do not reopen the identity-Sinkhorn screen

The corrected two-unit S1 identity-Sinkhorn continuation has already run and
returned `STOP`: neither unit achieved the required held-out field reduction.
It should not be inserted as another prerequisite or silently carried into the
budget foundation.

### Advance EMF without calling it a pass

Select EMF because it is the best-supported objective among the tested arms.
This is a resource-allocation decision, not a statistical promotion.

## 10. Next experiment: `CAP-EMF-1`

### 10.1 Scientific question

Can a stronger, spatially adequate raw-pixel EMF produce recognizable,
diverse CIFAR-10 automobiles with exactly one network call and no learned
encoder, when given the full rented-GPU training budget?

This is the next training experiment. There is no intervening local quality
rerun.

### 10.2 Frozen foundation proposal

| component | proposed value |
|---|---|
| data | all 5,000 CIFAR-10 automobile train images; 1,000 test images sealed |
| representation | raw RGB pixels; no VAE, perceptual encoder, teacher, or reference bank |
| inference | one direct endpoint call |
| objective | audited direct-`x` EMF, JVP-free |
| image tokens | patch 2, hence 256 tokens |
| trunk | width 512, depth 12, eight heads, long U-shaped skips |
| conditioning | AdaLN-Zero-style time/interval conditioning rather than two extra scalar tokens |
| local output | shallow convolutional pixel head/refiner; no learned external representation |
| optimizer | AdamW/Adam family, `1e-4`, betas `(0.9,0.95)`, zero weight decay |
| effective batch | 64, using accumulation if required |
| training | 160,000 updates / 10.24M examples / about 2,048 nominal class epochs |
| checkpoints | 20k, 40k, 80k, 120k, 160k |
| precision | BF16/AMP, with sensitive reductions and diagnostics in FP32 |
| correction | none during the foundation |

This architecture is a conservative synthesis, not a claimed reproduction:

- 256 tokens match the spatial token count of EMF's JiT pixel experiment;
- direct clean-image prediction and AdaLN-Zero follow JiT/EMF principles;
- U-shaped skips and a local output head use the existing CIFAR U-ViT evidence
  and directly target S3R's high-frequency defect; and
- width 512 keeps the model near the previously planned roughly 45--55M
  budget rather than jumping to a 131M DiT-B or an untested dual-level model.

All scientific knobs must be frozen before the cloud benchmark. The benchmark
may change only microbatch/accumulation, fused-kernel settings, and the selected
GPU to fit memory and dollars.

### 10.3 Corrected health protocol

The old outcome is not rescored. `CAP-EMF-1` should preregister:

1. raw endpoint moment, centered variance, and rank at every checkpoint;
2. EMA metrics only after its declared maturity point;
3. rank noncollapse using `final >= 0.8 * min(best, 1)`, never final/max alone;
4. LL/LH/HL/HH variance and rank, with high-high energy explicitly reported;
5. 512-sample frequent health clouds and a 2,048-sample checkpoint audit;
6. diagonal/interior EMF errors, pre-clip norms, clip fraction, and finite-rate;
7. exact inference-call counting; and
8. fixed uncurated train-only grids for catastrophic visual checks, never for
   test-set hyperparameter selection.

The 160k final checkpoint is primary. An intermediate checkpoint may terminate
the run only for a predeclared mechanical catastrophe such as NaNs, sustained
rank below 0.2 after adequate amplitude, or irrecoverable clipping. It is not a
new tuning ladder.

### 10.4 Cloud launch sequence

1. **Port and audit locally, without a quality run.** Test shape identities,
   direct-`x` EMF finite differences, one-call inference, restart determinism,
   EMA math, and the corrected gate.
2. **Spend at most $1 on an exact-code benchmark.** Benchmark the real model,
   data loader, AMP, optimizer, EMA, and checkpoint stack on available 4090/5090
   instances. Select by measured dollars per update.
3. **Project the complete cost.** Launch only if foundation plus evaluation
   stays below the reserved budget boundary.
4. **Train one full capability unit.** One strong unit is more informative for
   this proof of concept than two half-length units. It remains developmental.
5. **Evaluate once, after the frozen foundation completes.** Report uncurated
   images, KID, precision/recall, effective rank, multiscale spectrum,
   memorization, and one-call inference. FID against only 1,000 class-test
   images is report-only and not comparable to 50k-sample published FID.
6. **Fork only after capability.** If the foundation is recognizable,
   noncollapsed, and nonmemorizing, clone its exact model/optimizer/random
   state into a matched control and a protected formalization-derived
   correction continuation. Select the correction from already positive B1/B2
   evidence; do not revive the failed S1 Sinkhorn arm without new evidence.

### 10.5 No-loop rule

The next decision tree is intentionally short:

| outcome | action |
|---|---|
| cloud benchmark exceeds budget | reduce width once, not the patch grid or training objective; rebenchmark |
| EMF foundation is still unrecognizable/collapsed | stop the encoder-free one-step image claim for this budget |
| recognizable but detail-poor | record the proof of capability; a later project may test a dual-level pixel refiner |
| recognizable, diverse, nonmemorizing | run the matched correction fork |
| correction improves quality/coverage | promote the hybrid proof of concept |
| correction only lowers drift energy | retain the foundation result; do not claim image-quality gain |

There is no branch that sends us back to another 12,500-update S3R factorial.

## 11. Claim boundary for a favorable budget result

Even a strong result would support only:

> On a focused one-class 32-by-32 task, a raw-pixel, one-call EMF foundation
> can generate without a learned encoder; if the matched protected arm wins,
> a formalization-derived encoder-free drift can improve that foundation.

It would not establish ImageNet-scale encoder independence or reproduce the
original paper's full model. Those are later scaling questions.

## 12. Primary sources

- Local result artifacts:
  [`stage_pmf_r/runs`](encoder_independent_drifting/stage_pmf_r/runs/)
- Local S3/S3R diagnosis:
  [`EncoderIndependentS3FailureResearch.md`](EncoderIndependentS3FailureResearch.md)
- Local Sinkhorn S1 protocol and result artifact:
  [`EncoderIndependentSinkhornS1Protocol.md`](EncoderIndependentSinkhornS1Protocol.md),
  [`s1_v2_initial_two.json`](encoder_independent_drifting/stage_sinkhorn/s1_v2_initial_two.json)
- Li et al., [Trajectory Consistency for One-Step Generation on Euler Mean
  Flows](https://arxiv.org/html/2602.02571)
- Lu et al., [One-step Latent-free Image Generation with Pixel Mean
  Flows](https://arxiv.org/html/2601.22158v3)
- Zhang et al., [AlphaFlow: Understanding and Improving MeanFlow
  Models](https://arxiv.org/html/2510.20771)
- Li and He, [Back to Basics: Let Denoising Generative Models
  Denoise](https://arxiv.org/html/2511.13720)
- Yu et al., [PixelDiT: Pixel Diffusion Transformers for Image
  Generation](https://arxiv.org/abs/2511.20645)
- Karras et al., [Analyzing and Improving the Training Dynamics of Diffusion
  Models](https://arxiv.org/abs/2312.02696), for post-hoc EMA methodology

