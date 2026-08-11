# Local S3 Pixel MeanFlow Results

**Run completed:** 2026-08-03 03:48 local time  
**Verdict:** mechanically valid, scientifically failed S3 capability gate

## Question tested

Could a resource-scaled, raw-pixel pixel MeanFlow learn a recognizable,
diverse CIFAR-10 automobile generator with exactly one network evaluation,
without a VAE, pretrained feature encoder, teacher, reference bank, solver, or
classifier-free-guidance second pass?

This was a foundation test. It did not yet attach the proposed drifting or
Sinkhorn correction and was not designed as a paper-superiority comparison.

## Run integrity

Both predeclared units completed 60,000 updates and 960,000 training examples
(192 nominal class epochs). The official 5,000-image automobile training split
was the only optimization data. The 1,000-image official test split was not
accessed until both units had finished training.

| item | unit 700 | unit 701 |
|---|---:|---:|
| optimizer updates | 60,000 | 60,000 |
| examples seen | 960,000 | 960,000 |
| training time | 2 h 47 m | 2 h 32 m |
| peak allocated GPU memory | 3.48 GiB | 3.48 GiB |
| training parameters | 30,499,296 | 30,499,296 |
| inference parameters | 30,480,816 | 30,480,816 |
| inference NFE | 1 | 1 |

All eight 2k/10k/30k/60k checkpoint hashes, both result-shard hashes, the
profile hash, and the frozen source hash were verified. No restart occurred.
Both units used the same sealed evaluation noise, while their training streams
were independent.

## Final quantitative results

| report-only metric | unit 700 | unit 701 | interpretation |
|---|---:|---:|---|
| FID, 1,000 samples | 273.18 | 274.92 | extremely poor; absolute value is small-sample biased |
| KID | 0.3443 | 0.3474 | large discrepancy from test automobiles |
| precision | 0.003 | 0.002 | almost no samples lie in the estimated real feature manifold |
| recall | 0.000 | 0.000 | no measurable target coverage |
| precision/recall F1 | 0.000 | 0.000 | failed |
| effective rank | 2.50 | 2.63 | severe low-dimensional collapse |
| second-moment ratio | 0.407 | 0.348 | substantially under-dispersed |
| spectrum exponent | 4.20 | 4.12 | blurrier than real automobiles (about 3.59) |
| exact-near duplicate rate | 0.000 | 0.000 | not literal duplicate collapse |
| nearest-neighbour diversity | 0.529 | 0.533 | continuous variation remains, but not valid coverage |
| samples with any pixel outside [-1,1] | 4.1% | 1.5% | modest range leakage |

FID is included only as an internally matched diagnostic. With 1,000 samples
and 2,048-dimensional Inception features it is not comparable to published
50,000-sample FID. KID and precision/recall lead to the same failure verdict,
so that caveat does not change the conclusion.

## Visual result

- [Unit 700 uncurated grid](encoder_independent_drifting/stage_pmf/runs/pmf_s3_unit_700_uncurated.png)
- [Unit 701 uncurated grid](encoder_independent_drifting/stage_pmf/runs/pmf_s3_unit_701_uncurated.png)

Both grids show blurry colored automobile-like blobs. They do not contain
recognizable cars. Because the evaluation prior was paired, corresponding
cells can be compared directly: the two independent models produce very
similar geometry. On 256 paired outputs their mean cosine similarity is
0.986, with relative RMS difference 0.164. This is a replicated failure mode,
not a single unlucky seed.

The samples are not exact copies of one training image. However, the generated
sets claim only 263--295 distinct nearest augmented training examples out of
1,000 samples, and their median normalized nearest-training distances are
0.622--0.626. Blur and low-dimensional collapse make those memorization
statistics hard to interpret positively; they do not rescue the model.

## Checkpoint diagnosis

The following checkpoint comparison was performed after the run. It is a
diagnostic, not a predeclared selection rule.

| unit | step | FID | KID | precision | recall | effective rank | moment ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 700 | 2k | 350.15 | 0.4636 | 0 | 0 | 94.72 | 0.006 |
| 700 | 10k | 298.77 | 0.3689 | 0 | 0 | 51.63 | 0.136 |
| 700 | 30k | 270.64 | 0.3259 | 0.001 | 0 | 2.44 | 0.357 |
| 700 | 60k | 273.18 | 0.3443 | 0.003 | 0 | 2.50 | 0.407 |
| 701 | 2k | 351.09 | 0.4651 | 0 | 0 | 92.66 | 0.006 |
| 701 | 10k | 311.08 | 0.3863 | 0 | 0 | 51.65 | 0.108 |
| 701 | 30k | 266.20 | 0.3201 | 0.002 | 0 | 2.62 | 0.337 |
| 701 | 60k | 274.92 | 0.3474 | 0.002 | 0 | 2.63 | 0.348 |

At 2k--10k, outputs retain many tiny-variance directions but are close to zero
amplitude and far from the data. Between 10k and 30k, output magnitude becomes
more realistic while effective rank collapses. The 30k checkpoint has slightly
better FID/KID than the final checkpoint, but precision is still at most 0.2%
and recall remains zero. Therefore early stopping alone cannot make this a
successful generator.

## Training-dynamics diagnosis

The auxiliary instantaneous-velocity head behaved as intended: its median raw
error improved from roughly 0.23 early to 0.18 late in both units. The main
MeanFlow field did not remain stable. Its median raw error was best around
10k--20k and subsequently worsened; its upper tail and JVP magnitude grew
again after 30k. In the final 10k updates:

- unit 700 main raw-error median/p90 was 0.394/2.249;
- unit 701 main raw-error median/p90 was 0.562/2.895;
- unit 701 JVP RMS p90 reached 4.75;
- individual late minibatches reached raw error around 18--20.

The adaptive objective stays numerically near 2 because each of its two terms
has the form `loss / stopgrad(loss + 0.01)`. Its scalar value is therefore not
a useful convergence monitor. The unnormalized error, JVP tail, output rank,
and output moment are the important diagnostics.

## What succeeded

1. The experiment establishes a genuine raw-pixel, encoder-free training and
   generation path with exactly one inference network call.
2. The pMF/iMF identities, stopped JVP, clamp handling, full time triangle,
   source sealing, deterministic recovery, and evaluation isolation all worked.
3. The auxiliary marginal-velocity head learned stably and repaired the short
   learning sanity that the original boundary-only draft failed.
4. The failure reproduced across independent training units, yielding useful
   evidence about this architecture rather than a one-off crash.

## What failed

S3 required recognizable, noncollapsed, diverse automobiles in both units.
It failed all substantive quality/coverage criteria:

- images are not recognizable;
- recall is zero;
- precision is approximately zero;
- the final generated distribution has only about 2.5 effective dimensions;
- longer training trades variance growth for collapsed geometry rather than
  approaching the target distribution.

Consequently this foundation should not be promoted to S4, and no claim of an
encoder-independent competitive drifting model follows from this run.

## Most likely causes

1. **Conditional-mean blur.** Pixel MSE and velocity regression reward averages
   when one noise input remains compatible with multiple images. Published pMF
   relies heavily on perceptual losses to restore semantic/detail quality; this
   strict encoder-independent run intentionally removed them.
2. **Insufficient spatial inductive bias.** The local model has an 8-by-8 token
   grid (4-by-4 patches) and no convolutional refiner. It can model global color
   and silhouette more easily than localized automobile detail.
3. **Underpowered auxiliary separation.** The resource-scaled auxiliary head
   shares the complete backbone and differs only at its final projection. The
   released large pMF model gives the auxiliary velocity branch multiple
   unshared transformer layers.
4. **Uncontrolled interior dynamics.** The JVP/error tails begin growing after
   the early optimum even while the auxiliary FM task continues improving.
5. **No coverage-preserving objective.** Nothing in the present loss directly
   prevents the one-step endpoint map from concentrating variation in only a
   few directions.

## Recommended next move

Do not spend another full run merely changing the seed or stopping at 30k.
The next developmental experiment should change the mechanism while preserving
one-step, raw-pixel inference:

1. add train-only endpoint health monitoring (moment ratio, effective rank,
   JVP-tail quantiles) so collapse is detected before test evaluation;
2. use an FM-to-full-triangle curriculum: first learn the denoised/instantaneous
   field, then gradually introduce interior intervals rather than exposing the
   fragile self-referential JVP target at full weight from update one;
3. replace the final-only auxiliary head with a small genuinely unshared branch;
4. add encoder-free structural supervision such as a fixed multiscale
   Laplacian/wavelet, image-gradient, and local-patch loss;
5. add local spatial capacity through a lightweight convolutional refiner or a
   finer token grid, subject to a fresh cost preflight;
6. run a short checkpointed factorial first. Advance only an arm that improves
   endpoint amplitude without collapsing rank and lowers both diagonal and
   interior fixed diagnostics.

These changes target the observed failure directly. A pretrained perceptual
encoder would probably improve images faster, but it would abandon the central
encoder-independence question this experiment was designed to test.

## Audited artifacts

- `numerics/EncoderIndependentPMFS3Protocol.md`
- `numerics/EncoderIndependentPMFS3Audit.md`
- `numerics/encoder_independent_drifting/stage_pmf/runs/pmf_s3_unit_700.json`
- `numerics/encoder_independent_drifting/stage_pmf/runs/pmf_s3_unit_701.json`
- `numerics/encoder_independent_drifting/stage_pmf/s3_full_run.log`
