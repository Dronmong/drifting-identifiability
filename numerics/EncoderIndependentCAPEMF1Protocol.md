# CAP-EMF-1 — capability protocol for a one-call raw-pixel EMF foundation

**Status:** frozen run card; not implemented, not benchmarked, not run
**Date:** 2026-08-04
**Predecessor:** [`EncoderIndependentS3RResultsAndBudgetDecision.md`](EncoderIndependentS3RResultsAndBudgetDecision.md) §10
**Consumer:** [`AnchoredSelfFeatureDriftingSpecification.md`](AnchoredSelfFeatureDriftingSpecification.md) — CAP-EMF-1 is its precondition P1 and supplies its frozen feature trunk

---

## 1. The question

> Can a spatially adequate raw-pixel Euler Mean Flow model produce recognizable,
> diverse **unconditional CIFAR-10** images with **exactly one network call** and
> **no learned encoder**, given a full rented-GPU training budget?

### 1.1 Amended 2026-08-04: target and budget

Two changes, both driven by the stated priority that the mechanism must produce
**coherent images** and by the head-to-head deliverable in
[`EncoderIndependentHeadToHeadProtocol.md`](EncoderIndependentHeadToHeadProtocol.md).

| | was | now | why |
|---|---|---|---|
| target | 5 000 automobile images | **50 000, all classes** | 5 000 images at this horizon is ~4 096 epochs, which makes **memorization the likely route to coherent-looking samples**. It also makes the standard 50 k-sample FID protocol available, so the headline number is comparable to published CIFAR-10 results instead of a small-sample-biased number against 1 000 images. |
| width | 512 (63.5 M) | **384 (37.7 M)** | 37.7 M is essentially DDPM's 35.7 M and sits inside the 35–56 M band every strong CIFAR-10 model occupies. Width 512 measured **1.43× slower** for no capacity argument at 32×32: at a fixed GPU budget the extra parameters buy themselves out of ~30% of the images seen. |
| updates | 160 000 | **750 000** | 160 k was chosen by analogy to nothing in particular. DDPM used 102 M images on this dataset; 750 k × 64 = **48 M images (960 epochs)** is roughly half that at its parameter count — the closest a 48 h / $25 envelope gets to a known-good configuration. |
| precision | unspecified | **TF32 matmul, FP32 storage** | measured 1.23–1.30× on Ada, with FP32 inputs, outputs and accumulation. See §3.1 for why BF16 is rejected. |
| local difference | dense | **active rows only** | the two stopped evaluations were being run on rows whose quotient is multiplied by zero. Skipping them is mathematically exact and measured **1.24×**. |

**Combined: 2.25× more images seen per GPU-hour** than the configuration this
protocol was first written against — 0.987 s/update versus 2.223, measured on
the same card.

### 1.2 What this run is short of

Parameters are **not** the constraint. At 37.7 M we sit at published CIFAR-10
scale. Images seen is the constraint: DDPM used 102 M and EDM ~200 M, against
our 48 M. Any future budget should be spent on more images, not a bigger model.

One question, one unit, one evaluation. This is a **capability experiment**, not
a comparison and not a mechanism screen — S3R already made the mechanism choice
and is closed.

**No drifting correction is present.** Laplace, spectral and Sinkhorn terms are
all absent, so a failed foundation can never be mistaken for a failed
correction. The fork comes after (§10).

---

## 2. What S3R established, and what it left

S3R's one-unit screen selected direct-`x` EMF as the strongest of three
JVP-free/JVP mechanisms: raw second moment 0.495, centered variance 0.504, rank
ratio 1.661 with no collapse, half the runtime of continuous pMF. **No arm
passed its gate**, and that result is not rescored here.

Three of the four obstructions S3R diagnosed are addressed by this protocol:

| S3R obstruction | CAP-EMF-1 |
|---|---|
| patch 4 → only 64 image tokens | **patch 2 → 256 tokens** |
| 12 500 updates | **160 000 updates** |
| EMA immature at that horizon (28.7% initialization weight remaining) | 160 k updates ≈ **23 half-lives**; residual initialization weight ≈ 1×10⁻⁷ |
| rank-retention rule directionally wrong | corrected rule, §6.3 |

The fourth — **high-frequency deficit** — is the one this experiment actually
risks. S3R's EMF arm reached Haar variance ratios LL 0.509, LH 0.410, HL 0.512,
**HH 0.159**. Patch 2 and a local pixel head target it directly, but nothing
guarantees they fix it, and §7 makes HH an explicit gate rather than a footnote.

---

## 3. Frozen scientific configuration

Every value below is frozen before the cloud benchmark. The benchmark may
change **only** microbatch/accumulation split, fused-kernel settings, and the
selected GPU.

| item | value |
|---|---|
| data | **50 000 CIFAR-10 train images, all classes**; the 10 000 test images are sealed |
| representation | raw RGB 32×32; no VAE, encoder, teacher, reference bank, or perceptual loss |
| inference | one direct endpoint call, counted |
| objective | direct-`x` Euler Mean Flow, JVP-free (§4) |
| patch size | **2** → 16×16 = **256 image tokens** |
| trunk | **width 384**, depth 12, 8 heads, mlp ratio 4.0, long U-shaped skips |
| parameters | **37 726 863** (measured) |
| conditioning | **AdaLN-Zero** for absolute time and interval (§5.2) |
| output | shallow convolutional pixel head/refiner |
| auxiliary branch | **none** — EMF is JVP-free and needs no velocity branch |
| optimizer | AdamW, lr `1e-4`, betas `(0.9, 0.95)`, weight decay 0 |
| effective batch | 64 (microbatch × accumulation, split freely) |
| gradient clip | 10.0, with clip fraction logged |
| updates | **750 000** = 48 M examples = **960 epochs** |
| EMA | 0.9999 (108 half-lives; residual initialization weight 2.67e-33) |
| checkpoints | 100 k, 200 k, 400 k, 600 k, **750 k (primary)** |
| precision | **TF32 matmul, FP32 storage and accumulation** (§3.1) |

### 3.1 Precision: TF32, and why not BF16

The earlier draft specified BF16/AMP. That is **rejected**, for a numerical
reason rather than a performance one:

> The EMF local difference computes `(future − current) / δ` with `δ = 0.01`,
> from two evaluations of the *same network* at nearby inputs. It is a
> difference of nearly-equal quantities divided by a small number — the
> textbook setting for **catastrophic cancellation**. BF16 carries 2–3
> significant decimal digits, so with outputs of order 1 the difference of
> order 0.01 would be dominated by rounding noise.

TF32 reduces mantissa precision *inside the matmul only*. Inputs, outputs and
accumulation remain FP32, so the subtraction happens at full precision and the
cancellation problem does not arise. Measured **1.23–1.30×** on Ada, for a
one-line change and no numerical risk.

If a future run wants BF16 anyway, it must re-pass the first-order convergence
check of §8.1 with the local difference held in FP32.
| augmentation | horizontal flip only, recorded per example |
| correction | **none** |

The 160 k checkpoint is primary. An intermediate checkpoint may terminate the
run only for a predeclared mechanical catastrophe (§8.3). It is **never** a
tuning ladder and may not be selected as the result.

---

## 4. Objective

Direct-`x` EMF, ported verbatim from the audited S3R implementation
([`stage_pmf_r/objectives.py`](encoder_independent_drifting/stage_pmf_r/objectives.py)),
under this repository's data-at-zero / noise-at-one clock. For local step
`δ = 0.01`:

```
z'_{t−δ} = z_t + (δ/t)·( x̂_θ(z_t, t, 0) − z_t )
```

and, when `t − r > δ`, the detached target

```
x + (t − r − δ)·(t/r)·[ x̂_θ(z'_{t−δ}, t−δ, r) − x̂_θ(z_t, t, r) ] / δ
```

Short and diagonal intervals reduce to direct endpoint regression. The paper's
reversed-clock `1/(1−t_paper)²` weight becomes **`1/t²`**, and its `0.02` clamps
on `1−t_paper` and `1−r_paper` become clamps on local `t` and `r`.

**The numbered equations, not the arXiv HTML pseudocode, are the source of
truth** — the rendered pseudocode contains transcription errors. A float64
regression against the corresponding directional JVP is mandatory (§8.1).

Time sampling, diagonal fraction, and adaptive weighting are inherited
unchanged from the S3R developmental profile: logit-normal `(0.8, 0.8)`,
diagonal fraction 0.5, adaptive power 1.0, adaptive epsilon 0.01.

---

## 5. Architecture

### 5.1 Trunk

A U-ViT: encoder blocks 1–6, decoder blocks 7–12, each decoder block fusing the
reversed encoder skip through a `Linear(2·width, width)`. **Uniform width and
uniform token count throughout — there is no spatial bottleneck**, and no
section of this protocol or of ASFD may describe one.

### 5.2 Conditioning — AdaLN-Zero, and why it is not cosmetic

S3R prepends two conditioning tokens
(`torch.cat((time_token, interval_token, patches), dim=1)`,
[`stage_pmf_r/model.py:121`](encoder_independent_drifting/stage_pmf_r/model.py#L121)),
so its token sequence is `256 + 2`. CAP-EMF-1 replaces this with per-block
AdaLN-Zero modulation driven by the summed time and interval embeddings.

Two reasons, one of which is downstream:

1. AdaLN-Zero is the JiT/DiT-standard conditioning for direct-`x` pixel
   prediction and starts each block at identity, which S3R's clip-boundary
   behaviour (4.99% of updates at the 5% limit) suggests is worth having.
2. **ASFD extracts features from the image-token grid.** With AdaLN-Zero the
   grid is exactly `tokens`, with no slice; under S3R's scheme it is
   `tokens[:, 2:]`, and an off-by-two slice is invisible to a forward-parity
   test because a read-only hook genuinely does not change model outputs. The
   specification's §4.3 depends on this decision being recorded.

### 5.3 Local pixel head

A shallow convolutional refiner after unpatchify, targeting S3R's HH deficit
directly. It is part of the single inference call and adds no second call. It
introduces **no learned external representation** — it is a few convolutions on
the model's own pixel output.

### 5.4 Parameter budget

Estimated ≈ 60 M: 12 blocks × ≈ 3.15 M attention+MLP ≈ 37.8 M, 12 × 1.57 M
AdaLN-Zero modulation ≈ 18.9 M, 6 skip fusions ≈ 3.1 M, plus patch embed, head
and refiner.

**The port must report the exact count, and it is frozen at that value.** The
declared ceiling is **65 M**. If the cloud benchmark exceeds budget, S3R's
no-loop rule applies: **reduce width once** — never the patch grid, never the
objective, never the training horizon. A shared conditioning MLP with per-block
linear projections is the first reduction to try, since it removes most of the
18.9 M without touching capacity.

---

## 6. Train-only health, measured throughout

No test image, Inception feature, FID, KID, or human grid is used during
training. The test split is not instantiated by the training process.

### 6.1 Endpoint health

At every health interval and checkpoint, on **sealed train-only noise**:

- raw second-moment ratio **and** centered-variance ratio — reported together,
  because a nonzero constant image clears a second-moment threshold with zero
  diversity;
- raw effective rank, interpreted jointly with amplitude — a high rank at
  moment ratio 0.006 is not diversity;
- Haar LL/LH/HL/HH variance and rank, with **HH energy explicitly reported**.

Cloud sizes: **512** fixed samples for frequent checks, **2 048** for checkpoint
audits. S3R's 64 was adequate for collapse detection and far too noisy for a
threshold decision.

### 6.2 EMA maturity

EMA metrics are reported from **40 000 updates** onward — 5.8 half-lives, at
which the initialization retains 0.3% weight. Before that they are logged and
explicitly marked immature. Raw (non-EMA) checkpoints are retained alongside
EMA at every checkpoint step.

Parameter snapshots sufficient for **post-hoc EMA reconstruction** (Karras et
al.) are stored at every checkpoint. This is cheap insurance and it also serves
ASFD §11.1's fallback, which wants a trunk checkpoint that is *not* the fork
point.

### 6.3 Corrected rank rule

```
R_final  ≥  0.8 · min(R_best, 1)
```

never `final / max`. S3R's rule called EMF's move from an over-dispersed
noise-like ratio of 4.056 down toward the target ratio of 1.0 a "0.410
retention" failure — directionally wrong. The corrected rule still rejects
pMF's 0.349.

### 6.4 Optimization health

Diagonal and interior EMF errors; pre-clip gradient norm distribution; clip
fraction; non-finite rate; wall time; peak memory; **exact inference forward
count**; examples seen; EMA length in both updates and images.

---

## 7. Capability gate

Two stages. The training-time stage uses no test data; the sealed stage runs
**once**, after the 160 k checkpoint is frozen.

### 7.1 Train-only preconditions

All required, all measured at 160 k on 2 048 samples:

| # | requirement | threshold |
|---|---|---|
| H1 | raw second-moment ratio | ≥ 0.80 |
| H2 | raw centered-variance ratio | ≥ 0.80 |
| H3 | rank noncollapse | `R_final ≥ 0.8·min(R_best, 1)` |
| H4 | **HH Haar variance ratio** | **≥ 0.50** |
| H5 | LH and HL Haar variance ratios | ≥ 0.60 each |
| H6 | non-finite update rate | 0 |
| H7 | clip fraction over the final 20 k updates | < 5% |
| H8 | inference forward count | exactly 1 |

**H4 is the discriminating threshold.** S3R's EMF arm reached 0.159. It is set
at 0.50 because ASFD's feature qualification gate G7 requires per-band
sensitivity `ρ_b ∈ [0.25, 4.0]` in *every* band, and a trunk trained to a model
that cannot render diagonal detail is unlikely to encode it. **If H4 fails, the
foundation may still be a valid capability result, but ASFD's G7 is at high risk
and that must be recorded before the fork is designed.**

### 7.2 Sealed evaluation, run once

Runs once, after the checkpoint is frozen and hashed.

**FID follows the standard CIFAR-10 protocol: 50 000 generated samples against
the 50 000 training images.** This is the whole reason the target changed — the
headline number is directly comparable to published CIFAR-10 unconditional FID
rather than an indicative figure against 1 000 images. Generating 50 k samples
from a one-call model costs a few minutes.

The 10 000 sealed test images are opened only here, and only for the held-out
checks. Report:

- **FID-50k against the training set** — the comparable headline number;
- FID against the sealed 10 k test set — honest held-out cross-check;
- KID with paired without-replacement subsampling;
- fixed-seed **uncurated** one-call sample grids, each shipped **beside its
  nearest-training-image grid** so coherence cannot be achieved by copying;
- precision/recall and density/coverage;
- raw-pixel effective rank and multiscale spectrum;
- duplicate rate and nearest-training-image distance distribution.

### 7.3 Verdict

| verdict | condition |
|---|---|
| **PASS** | all of H1–H8, uncurated grids show recognizable automobiles, nontrivial recognizability, no memorization veto |
| **PASS (detail-poor)** | as above but H4 fails |
| **FAIL** | otherwise |

**No metric may select a post-hoc checkpoint on the test allocation.** The 160 k
checkpoint is the result.

---

## 8. Execution order

### 8.1 Local port and audit — no quality run

The 6 GiB laptop cannot train this model, and will not try. It must verify:

1. shape identities across patch 2 / width 512 / AdaLN-Zero;
2. **direct-`x` EMF finite difference against the directional JVP in float64**;
3. one-call inference and an exact forward counter;
4. restart determinism and every RNG stream;
5. EMA mathematics and post-hoc snapshot reconstruction;
6. the corrected rank rule and Haar energy conservation;
7. **feature-tap parity** — extracting blocks 3/6/9/12 leaves ordinary outputs
   bit-identical, and the extracted grid reshapes to exactly 16×16 with no
   conditioning-token slice (§5.2);
8. exact parameter count against the 65 M ceiling;
9. a reduced-microbatch throughput and memory probe for extrapolation only.

Item 7 is included now, not deferred, because retrofitting hooks after the run
would change the trunk's source hash and break exactly the kind of binding that
blocked B2.5.

### 8.2 Cloud benchmark and cost gate

Spend **at most $1** benchmarking the *exact* code — real model, loader, AMP,
optimizer, EMA and checkpoint stack — on available 4090/5090 instances. Select
by measured dollars per update. Project the full 160 k cost plus evaluation, and
**launch only if it stays under the reserved budget boundary**.

### 8.2.1 Budget envelope — 48 hours, $25

The frozen configuration is sized to this envelope, not the other way round.
All timings are measured on the RTX 4050 laptop at production shape
(`micro_batch 16`, TF32, 3.25 GiB) and scaled by 5–7× for a rented 4090.

| item | measured / projected |
|---|---|
| local, 4050 | **0.987 s/update**, 205.7 h for 750 k updates |
| 4090 at 6× | **≈ 0.165 s/update → ≈ 34 h** |
| 4090 at 5× (pessimistic) | ≈ 41 h |
| 4090 at 7× (optimistic) | ≈ 29 h |
| rental, $0.40/h | **≈ $14** |
| rental, $0.50/h | ≈ $17 |
| benchmark + evaluation | ≈ $2 |
| **total** | **≈ $16, ≈ 36 h wall** |

Headroom against the envelope: ~12 h and ~$9 even at the pessimistic scaling.

**How the envelope was met.** Three measured changes, together **2.25× more
images seen per GPU-hour** (0.987 s/update against 2.223):

| change | measured | note |
|---|---:|---|
| active-row gather in the local difference | **1.24×** | the two stopped evaluations were running on rows whose quotient is multiplied by zero; over half of every batch |
| TF32 matmuls | **1.25×** | FP32 storage and accumulation preserved |
| width 512 → 384 | **1.43×** | 37.7 M ≈ DDPM's 35.7 M; the extra width bought itself out of ~30% of the images |

**What was rejected, and why.** A 5090 gives ~1.3× in FP32 but ~2.5× in BF16,
at typically 1.5–2× the rental rate — so it only pays off with BF16, and BF16
is ruled out by §3.1's cancellation argument. Under FP32/TF32 a 5090 is poor
value and the 4090 is the right card.

**Budget stop rule, predeclared.** If wall clock or budget is exhausted before
750 k, **the last completed checkpoint is the result** and the fact is recorded
in the artifact. This is a budget stop, not a metric selection: no checkpoint
may ever be chosen by looking at its numbers.

**If a second envelope becomes available, spend it on images seen, not on a
bigger model.** At 37.7 M we are at published CIFAR-10 parameter scale; at 48 M
images we are at roughly half DDPM's 102 M and a quarter of EDM's ~200 M. That
is the axis this run is short on.

### 8.3 The run

One full capability unit. Developmental scope. Early termination only for a
predeclared mechanical catastrophe: non-finite loss, sustained rank below 0.2
after adequate amplitude, or irrecoverable clipping. Nothing else stops it, and
nothing about it is tuned mid-flight.

---

## 9. Artifact and source discipline

**Hash an explicit dependency list, never a directory glob.**

This is the direct lesson of the B2.5 blocker
([`EncoderIndependentB25ResumeAudit.md`](EncoderIndependentB25ResumeAudit.md)
§7.1): `b1_freeze.py` globs `PACKAGE.glob("*.py")` and
`(PACKAGE / "tests").glob("*.py")`, so two unrelated test files added months
later permanently invalidated a completed confirmation, and the recorded bytes
proved unrecoverable. `stage_cap` will enumerate its dependencies explicitly, as
`stage_b25/artifacts.py:_DEPENDENCIES` already does for its parent package.

Every artifact records source hashes, environment, CUDA/PyTorch versions,
configuration payload, seed derivation manifest, data provenance, parameter
count, costs, and thresholds. Checkpoints record unit, step, protocol hash,
state dictionary and SHA-256, and loading rejects any mismatch. Runners refuse
to overwrite.

---

## 10.1 The matched budget for the drifting arm

The head-to-head comparison matches on **examples seen, not optimizer
updates**. EMF sees 64 examples per update; a drifting field needs a cloud of
order 256. Matching updates would hand drifting **four times** the data
exposure.

| arm | per update | updates | examples |
|---|---:|---:|---:|
| CAP-EMF-1 | 64 | 750 000 | 48 000 000 |
| base drifting, cloud 256 | 256 | **187 500** | 48 000 000 |

Both arms therefore see the same data the same number of times, on the same
trunk, against the same evaluation. Where the two cannot be matched — the batch
structure itself — both figures are reported.

## 10. Interface to ASFD

If the verdict is PASS, the foundation supplies:

| item | value |
|---|---|
| `θ*` | EMA weights at 160 k |
| feature taps | encoder block 3, encoder block 6, decoder block 9 (post-fusion), decoder block 12 (pre-`final_norm`) |
| token grid | exactly 256 tokens, 16×16, **no conditioning-token slice** |
| descriptor | 2×2 average pool → 8×8 local, plus global channel mean and std = 66 vectors per level |
| fallback trunks | 80 k and 120 k checkpoints, for ASFD §11.1 decorrelation |
| recorded risk | the measured HH ratio, as the leading indicator for ASFD's G7 |

The fork clones the exact model, optimizer, EMA and RNG state into matched
arms. Correction selection comes from already-positive B1/B2 evidence; the
failed S1 Sinkhorn arm is not revived without new evidence.

---

## 11. Claim boundary

A favourable result supports only:

> On unconditional CIFAR-10 at 32×32, a raw-pixel, one-call EMF foundation can
> generate recognizable, diverse images without a learned encoder, at a FID
> reported under the standard 50 k protocol.

Because the reference set and sample count now follow the standard protocol,
the FID **is** quotable next to published CIFAR-10 numbers — with the compute
difference stated, since 320 k updates at batch 64 is a fraction of what the
strongest published results use.

It does **not** establish ImageNet-scale encoder independence, reproduce the
drifting paper's complete model, or demonstrate that any correction helps. It
is one unit: developmental scope, no replication claim.

---

## 12. Primary sources

- Liu et al., [Euler Mean Flow](https://arxiv.org/html/2602.02571) — Equations 15–19, Appendix C.1, pixel JiT experiment
- Li and He, [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/html/2511.13720) — direct clean-image prediction
- Lu et al., [Pixel Mean Flows](https://arxiv.org/html/2601.22158v3) and [official code](https://github.com/Lyy-iiis/pMF)
- Zhang et al., [AlphaFlow](https://arxiv.org/html/2510.20771) and [official implementation](https://github.com/snap-research/alphaflow)
- Yu et al., [PixelDiT](https://arxiv.org/abs/2511.20645) — local pixel modelling
- Karras et al., [Analyzing and Improving the Training Dynamics of Diffusion Models](https://arxiv.org/abs/2312.02696) — post-hoc EMA
- Local: [`EncoderIndependentS3RResultsAndBudgetDecision.md`](EncoderIndependentS3RResultsAndBudgetDecision.md), [`EncoderIndependentS3FailureResearch.md`](EncoderIndependentS3FailureResearch.md), [`AnchoredSelfFeatureDriftingSpecification.md`](AnchoredSelfFeatureDriftingSpecification.md)
