# Head-to-head: our mechanism versus base drifting

**Status:** design; the drifting arm is not built
**Date:** 2026-08-04

## 1. The deliverable

One figure, two rows, one instrument.

| row | model | what it is |
|---|---|---|
| **1** | **ours** | one-call EMF foundation + self-feature drifting correction, fully trained |
| **2** | **base drifting** | the drifting objective on raw pixels — **no encoder, no decoder, no VAE, no teacher** |

Each row reports **FID**, KID, precision/recall, and a fixed-seed **uncurated**
sample grid for visual judgement. Nothing is curated, and every sample grid
ships beside its nearest-training-image grid.

The comparison is only worth making if row 1 produces coherent images. That is
the primary risk and it is gated before row 2 is trained (§6).

---

## 2. Why base drifting has to be re-trained

The repository already contains a raw-pixel drifting implementation —
[`EncoderIndependentB3Results.md`](EncoderIndependentB3Results.md). It is the
right *object* and the wrong *everything else* for this figure:

| | B3 | needed here |
|---|---|---|
| data | CIFAR-10, all classes | CIFAR-10, all classes ✅ |
| generator | `OneStepGenerator`, 147 k–3.86 M | the **63.5 M CAP trunk** |
| budget | 30 000 updates | 20.48 M examples matched |
| instrument | CIFAR test / CINIC-10 | standard 50 k FID protocol |

B3 also settled the question that would otherwise make matching architecture
look risky: **a 26× parameter increase changed drifting's recall by nothing**
(0.0000 → 0.0000). So putting drifting on the same 63.5 M trunk costs little and
removes the confound that forced every earlier drifting comparison to be hedged.

---

## 3. What "evenly matched" means here

Everything that can be held identical is held identical:

| held identical | value |
|---|---|
| trunk | CAP U-ViT, patch 2, width 512, depth 12, 63 548 687 parameters |
| data | the 50 000 CIFAR-10 training images |
| augmentation | horizontal flip only, same stream |
| examples seen | **20 480 000** |
| optimizer family | AdamW, `1e-4`, betas `(0.9, 0.95)`, weight decay 0 |
| gradient clip | 10.0 |
| EMA | 0.9999 |
| sampler | exactly **one** network call, counted |
| evaluation | identical reference sets, sample counts, estimators, seeds |
| precision | FP32 both arms |

### 3.1 The one thing that cannot be matched, and how it is handled

EMF consumes 64 examples per update; a drifting field needs a cloud of order
256. So updates and examples cannot both be matched.

**Matched on examples seen.** Matching updates instead would hand drifting
**four times** the data exposure:

| arm | per update | updates | examples |
|---|---:|---:|---:|
| ours | 64 | 320 000 | 20 480 000 |
| base drifting | 256 | **80 000** | 20 480 000 |

Both optimizer-update counts are reported alongside, so a reader who prefers
the other convention can see exactly what it would change.

### 3.2 What base drifting is, precisely

The Algorithm 2 bi-softmax mean-shift field on **raw pixels**, with the R11
scalar second-moment teacher correction, `smooth_laplace` kernel, target-ESS
bandwidth calibration, `paper` direction, RMS normalization — the configuration
B3 froze, moved onto the CAP trunk and this data.

**No encoder. No decoder. No VAE. No pretrained teacher. No perceptual loss.**
That is the honest "base" the encoder-independence question is about, and it is
what the paper's own text says did not work at ImageNet scale without a feature
extractor. Row 2 is therefore expected to be weak; the point of the figure is
that the weakness is measured on matched axes rather than asserted.

---

## 4. Evaluation

Identical for both rows, run once per row after its checkpoint is frozen and
hashed.

**Headline FID follows the standard CIFAR-10 protocol**: 50 000 generated
samples against the 50 000 training images. This is the reason the target moved
off a single class — the number is quotable next to published CIFAR-10
unconditional FID, with the compute difference stated. Generating 50 k samples
from a one-call model takes minutes.

| quantity | detail |
|---|---|
| **FID-50k** | 50 000 samples vs 50 000 train images — the headline |
| FID held-out | vs the 10 000 sealed test images |
| KID | paired without-replacement subsampling, unbiased at this size |
| precision / recall | Kynkäänniemi, fixed manifold, paired indicators |
| density / coverage | reported |
| effective rank | raw pixels |
| multiscale spectrum | Haar LL/LH/HL/HH |
| **memorization** | duplicate rate, nearest-training-image distance distribution |
| inference cost | forward count, asserted **= 1** |

### 4.1 Sample grids

Fixed seeds, declared before generation, **uncurated**. Each row shows:

1. a grid of one-call samples;
2. **directly beside it, the nearest training image to each sample.**

The second grid is not decoration. At 410 epochs a model that memorizes
produces beautiful samples, and a reader cannot tell the difference from the
first grid alone.

---

## 5. What the figure can and cannot say

**Can:** on unconditional CIFAR-10 at 32×32, with identical architecture, data,
data exposure, optimizer and evaluation, our mechanism reaches FID *x* and base
drifting reaches FID *y*, with these samples.

**Cannot:**

- that the paper's *complete* model fails — row 2 is the encoder-free objective,
  which is the comparison the encoder-independence question asks, not the
  paper's full system with its pretrained MAE and multi-scale features;
- anything about ImageNet or higher resolution;
- a replication claim — one training unit per row;
- a claim that our FID is competitive with published state of the art, unless
  it is, and then only with the compute difference stated.

---

## 6. Order, gates, and cost

Row 1 is built from the foundation the repository already has ported; row 2 is
new code. The gate between them exists because **if row 1 is not coherent there
is no figure**, and training row 2 first would spend a day proving something
about a comparison that cannot be made.

| step | what | cost (4090 est.) | gate |
|---|---|---:|---|
| 1 | CAP-EMF-1 foundation, 320 k updates | 29–40 h | capability gate §7 of the CAP protocol |
| 2 | sealed evaluation + sample grids | minutes | **are the images coherent?** |
| 3 | ASFD correction fork | 12–16 h | Stage D gate |
| 4 | base drifting, 80 k updates matched | measure first | none — it is the reference |
| 5 | one evaluation pass, both rows | minutes | — |

Local measurement: **200.3 h on the RTX 4050** for step 1 at production shape,
4.35 GiB, 2.253 s/update. Scaling by FP32 throughput and memory bandwidth gives
5–7× on a 4090.

**Step 4's cost is not yet measured** and must be before it is scheduled — a
256-sample cloud with a full backward on a 63.5 M trunk is a materially
different shape from anything probed so far. That measurement is the first
thing the drifting arm's preflight must produce.

Rough total: **45–60 h, on the order of $20–40.**

### 6.1 The honest risk

Step 2 is the real gate. 320 k updates at batch 64 is a fraction of what the
strongest published CIFAR-10 results use, and pMF reports that MSE-only
training is much worse than perceptually supervised training — a gap we are
choosing to accept, because importing a perceptual loss would abandon the
encoder-independence question this whole program is about.

So "coherent" may land anywhere between *recognizable object-like images with
visible artifacts* and *genuinely good samples*. If it lands below
recognizable, the figure does not exist and the correct response is to say so
rather than to ship row 2 against a broken row 1.

---

## 7. Build order from here

1. the sealed evaluator, hashed into the CAP preflight **before** the run, so it
   cannot be tuned against training curves;
2. the $1 cloud benchmark on exact code; project cost; authorize;
3. the foundation run;
4. **the coherence gate** — and an honest call at it;
5. the ASFD correction, then the matched drifting arm;
6. the figure.
