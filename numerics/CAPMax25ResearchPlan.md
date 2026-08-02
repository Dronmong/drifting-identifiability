# CAP-Max-25: budget-maximized encoder-free capability experiment

## Purpose and claim boundary

This document records the cloud-scale analysis that followed CAP-1R and
clarifies exactly which version of the encoder-dependence problem the proposed
experiment addresses.

The original drifting paper defines its field in raw data space, but explains
that high-dimensional image generation depends strongly on the quality of the
feature extractor.  Its ImageNet experiments use pretrained self-supervised
encoders, and the authors report that they were unable to make their method
work on ImageNet without a feature encoder.  They attribute this to the kernel
failing to express useful sample similarity when all raw or latent samples are
far apart.

CAP-Max-25 attacks the same practical obstruction, but it is important not to
overstate the result:

1. A successful flow-only foundation proves that a strong pixel-space
   generator can work without a learned training encoder.
2. A protected B1/B2 arm that beats its matched flow-only control shows that
   formalization-derived drifting geometry can improve an encoder-free
   generator.
3. It does **not**, by itself, prove that the original paper's drifting
   objective can train a generator from noise without an encoder.  The primary
   objective remains flow matching and B1/B2 are protected auxiliary terms.

The correct description of a positive result is therefore:

> An encoder-free pixel generator with formalization-derived drifting
> corrections, not a reproduction of the paper's pure drifting model without
> an encoder.

An exact test of the stronger claim would require a matched implementation of
the paper's full raw-space drifting training objective, trained without a
learned encoder.  Earlier B3 experiments tested a smaller raw-pixel one-step
proxy and failed even at matched capacity; they were not a complete
implementation of the published method.  That exact arm remains a separate,
higher-risk experiment.

## Why a compact target is the correct first target

The proposed target is the 5,000-image CIFAR-10 automobile training class,
with the 1,000 automobile test images sealed until final evaluation.  This is
not intended to demonstrate unrestricted image generation.  It is a focused
test of whether learned semantic features are strictly necessary for useful
drifting-derived geometry in a controlled image domain.

A compact target helps in four ways:

- **Lower target entropy.** The generator need not allocate capacity across
  ten unrelated semantic classes or ImageNet's thousand classes.
- **More meaningful fixed geometry.** Within one class at 32 by 32 resolution,
  raw and fixed multiscale distances have a better chance of tracking shape,
  pose, color, and coarse layout.  They do not have to discover that two
  semantically related objects with very different pixels belong together.
- **More optimization per target mode.** Every update trains the same target
  distribution rather than splitting a limited compute budget across many
  labels.
- **Clearer visual assessment.** Recognizability, diversity, collapse, and
  memorization can be inspected directly in uncurated automobile grids.

The target should not be made much smaller than one complete CIFAR class.
Using only hundreds or one or two thousand images would increase the chance of
memorization, destabilize held-out distribution metrics, and make success less
credible.  Five thousand training images plus a sealed 1,000-image test set is
a useful compromise between focus and statistical legitimacy.

The strong-model/small-target principle therefore means a strong, appropriate
architecture aimed at one complete and diverse class.  It does not mean an
arbitrarily large model aimed at a hand-curated micro-dataset.

## Correction to the original CAP-1R compute allocation

CAP-1R proposed 100,000 foundation updates at batch 40:

```
100,000 * 40 / 5,000 = 800 nominal dataset presentations.
```

That is shorter than the strongest relevant references after accounting for
dataset size and batch size:

- the official CIFAR U-ViT configuration presents about 1,280 nominal epochs;
- Meta's official pixel CIFAR flow-matching result reports its best result at
  about 1,800 epochs;
- the winning Curriculum Sampling checkpoint corresponds to about 2,048
  nominal epochs.

At batch 64 on 5,000 images, 160,000 updates provide 2,048 nominal
presentations.  The exposure-normalized version of the curriculum paper's
60,000-update switch is 96,000 updates.  This scaling is a reasoned transfer,
not a theorem that 96,000 is optimal, so intermediate checkpoints must be
preserved.

The revised foundation is:

| Item | Frozen value |
|---|---:|
| Batch size | 64 |
| Foundation updates | 160,000 |
| Curriculum | updates 0--95,999: `logit(t) ~ Normal(-0.8, 1)` |
| Final time law | updates 96,000--159,999: uniform |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| EMA | `0.9999` |
| Checkpoints | 40k, 80k, 100k, 120k, 140k, 160k |
| Path | pixel-space linear flow, velocity prediction |
| Pairing | exact fixed-multiscale minibatch OT |

## The strongest staged experiment

### Stage 0: exact-code cloud benchmark

Benchmark the complete training code, not a synthetic transformer loop, on an
RTX 4090 and RTX 5090 for a short fixed interval.  Measure ordinary foundation
updates and protected correction events separately.  Include data loading, OT
assignment, AMP, optimizer, EMA, logging, and checkpoint costs.

Current RunPod listings put the RTX 4090 at $0.69/hour and the RTX 5090 at
$0.99/hour.  The 5090 is cheaper per update only if it is more than

```
0.99 / 0.69 = 1.435
```

times as fast on the exact workload.  Select the GPU by measured dollars per
update, not peak specifications.

### Stage 1: convolutional reliability anchor

Train an approximately 50--55M parameter pixel U-Net using the same data,
path, pairing, timestep curriculum, optimizer, EMA, and evaluation protocol.
The official Meta flow-matching example reports CIFAR-10 FID 2.07 with a
pixel-space unconditional U-Net at batch 64, EMA, skewed time sampling, and a
50-NFE second-order solver.  This is the most direct evidence that the
encoder-free pixel-flow mechanism can generate good CIFAR images.

The U-Net is a systems and capability control.  If it cannot generate
recognizable, diverse automobiles, stop before blaming the transformer or
drifting correction: the likely problem lies in the shared data, path,
optimization, or evaluation implementation.

### Stage 2: transformer candidate

Train the official-scale CIFAR U-ViT-S/2 topology:

- 32 by 32 RGB pixels;
- patch size 2 and 256 image tokens;
- width 512;
- depth 12, with 13 executed blocks in the U-shaped construction;
- eight attention heads;
- long U-shaped skip connections;
- convolutional output head;
- approximately 45M parameters;
- batch 64 and 160,000 updates.

This is a stronger and more directly sourced candidate than CAP-1R's 26.1M,
width-384 adaptation.  Width 768 or a roughly 130M paper-sized DiT is not
recommended: it increases cost, auxiliary-trajectory memory, and overfitting
risk without direct evidence that such scale helps a 5,000-image target.

### Stage 3: matched formalization-derived fork

Only activate this stage after a foundation passes the preregistered
recognizability, diversity, effective-rank, clipping, and memorization gates.
Clone the exact model, EMA, optimizer, data cursor, and random streams.

- `control`: 16,000 additional flow-only updates;
- `protected`: the identical 16,000 flow updates plus protected B1/B2 every
  tenth update.

At batch 64, 16,000 updates represent approximately 205 nominal dataset
presentations.  This matches the approximately 200 presentations in the old
25,000-update, batch-40 continuation while saving compute.

Run the fork on U-ViT if U-ViT passes.  If U-ViT fails and the U-Net passes,
run it on the U-Net.  If both pass and the measured projection remains within
budget, applying the fork to both provides a valuable architecture-transfer
check.

The protected arm may be promoted only through a paired comparison with the
flow control.  Lower drift energy alone is insufficient; image quality,
coverage, effective rank, and memorization vetoes remain binding.

## Budget and launch gate

| Phase | Maximum allocation |
|---|---:|
| Environment plus 4090/5090 benchmark | $1 |
| U-Net anchor | $4 |
| U-ViT foundation | $9 |
| Matched continuation fork | $5 |
| Final sampling and evaluation | $2 |
| Recovery reserve | $4 |
| **Total** | **$25** |

Before a long launch, use the measured step times to project the complete
cost.  Do not launch if the main experiment projects above $21; the final $4
is reserved for interruption, checkpoint recovery, or a failed cloud
instance.  Use a single GPU, BF16 mixed precision with FP32 sensitive
calculations, fused AdamW, scaled-dot-product attention, RAM-cached data,
atomic checkpoints, and frequent off-instance artifact copies.

## Why this is a real attack on encoder dependence

The experiment removes every learned feature representation from training:

- no pretrained image encoder;
- no VAE encoder or latent tokenizer;
- no perceptual loss;
- no learned OT metric;
- no evaluation-network feature is allowed to influence training,
  checkpoint activation, kernel bandwidth, or sample updates.

The only geometry beyond pixels consists of fixed average-pooling/DCT-style
multiscale maps and analytically specified kernels.  These are deterministic
hand-designed operators, not learned semantic encoders.  Thus a protected-arm
improvement would show that a useful local drift signal can be built without
outsourcing similarity to a pretrained network.

The compact one-class target is central to that test.  It asks a deliberately
attainable first question:

> Can an encoder-free kernel express enough geometry to improve generation on
> a focused but nontrivial image distribution?

A positive result would justify scaling to additional classes and resolutions.
It would not yet establish that the same fixed geometry replaces encoders on
ImageNet.

## Interpretation matrix

| Outcome | Honest conclusion |
|---|---|
| U-Net and U-ViT both fail | Shared training stack or budget is inadequate; no conclusion about B1/B2 |
| U-Net works, U-ViT fails | Pixel flow works; the transformer recipe is the bottleneck |
| Both foundations work, protected arm loses | Encoder-free generation is viable, but current drifting corrections do not improve it |
| Protected arm lowers drift only | The formal mechanism is active but not the image-quality bottleneck |
| Protected arm improves quality and preserves coverage/rank | Strong evidence that formalization-derived drifting geometry benefits an encoder-free generator on a compact target |
| A future exact raw-drifting arm works from noise | Direct evidence for the stronger original target: drifting without a learned encoder |

## Primary references

- Lee and Chun, *Generative Modeling via Drifting*, especially Sections 3.3,
  3.4, Table 3, and the statement that ImageNet did not work without a feature
  encoder: `papers/2602.04770v2.pdf`.
- Meta, [official Flow Matching image example](https://github.com/facebookresearch/flow_matching/tree/main/examples/image).
- Bao et al., [official CIFAR U-ViT configuration](https://github.com/baofff/U-ViT/blob/main/configs/cifar10_uvit_small.py).
- Tong et al., [OT-CFM](https://arxiv.org/abs/2302.00482).
- [Curriculum Sampling](https://arxiv.org/html/2603.12517).
- Wang et al., [Patch Diffusion](https://arxiv.org/html/2304.12526), retained as a follow-up rather than mixed into the first cloud run.
- RunPod, [current GPU pricing](https://www.runpod.io/pricing).

