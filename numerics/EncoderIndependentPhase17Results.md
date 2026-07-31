# Encoder-Independent Kernel Drifting — Phase 17 results

## A pretrained encoder is catastrophically worse — and an *untrained* one is the best geometry tested

*Runner: `run_phase17.py`. Artifact: `phase17.json` (+ `.sha256`). 3
geometries × 2 seeds × 30 000 steps, CIFAR-10 at 32×32, target ESS 0.9, R11
on every arm, FID at 512 samples, GPU throughout. Read alongside
`EncoderIndependentLadderReadingGuide.md`, whose decision tree was fixed
before these numbers existed.*

---

## 0. The ladder

| geometry | FID | ED² | **tail** | 2nd moment | minutes |
|---|---:|---:|---:|---:|---:|
| **random ResNet18** *(untrained)* | **226.17** | **0.2471** | 0.1273 | 1.018 | 87.7 |
| raw pixels *(encoder-free)* | 234.72 | 0.2683 | 0.1547 | 0.947 | 24.0 |
| **pretrained ResNet18** | **373.19** | 0.5596 | **0.4853** | 0.978 | 83.1 |

*(floor 70.89, bar 248.08, noise ceiling ~430; real data tail ≈ 0.13)*

Paired within-seed differences, as the reading guide prescribed:

| contrast | paired mean | per-seed |
|---|---:|---|
| **pretrained − raw** | **+138.47** | +122.0, +155.0 |
| pretrained − random | +147.01 | +128.6, +165.4 |
| random − raw | −8.55 | −6.6, −10.5 |

**Both metrics agree on the ranking**, both seeds agree in sign on every
contrast, and the pretrained gap is **5× the 25.6-FID threshold** two seeds
can resolve. This is decision-tree branch (b), and it is not a marginal call.

---

## 1. Why this is credible, and why the obvious objection fails

The natural first objection is that the encoder arm was mis-calibrated —
that L2-normalized features on a unit sphere need a different operating point
than the ESS-0.9 rule provides.

**The `random_resnet` arm controls for exactly that.** It is the *same
architecture*, the *same L2 normalization*, the *same 128px input*, the *same
ESS-0.9 calibration procedure*, and the *same paired seeds*. It differs from
`pretrained_resnet` in **the weights alone** — and it is the **best geometry
tested**, beating raw pixels by 8.55 FID paired.

So the failure is specific to *pretraining*, not to the encoder pathway, the
normalization, or the calibration. That is the strongest control available
and it was in the design.

---

## 2. The mechanism signature

The tail column is the tell. Real CIFAR sits near 0.13; raw and random arms
land at 0.155 and 0.127. **The pretrained arm reaches 0.4853 — 3.7× the
data's** — while its second moment is a healthy 0.978.

So the pretrained-feature field is not collapsing the cloud; it is **injecting
high-frequency content**. The samples have the right overall scale and
grossly too much energy in the trailing directions.

That has a principled reading, and the research plan anticipated it in §2:
a pretrained semantic encoder is *trained to be invariant* to texture, colour
and position — precisely the details that make an image look real. Matching
such features does not constrain those details, so the field is free to fill
them with anything, and it fills them with noise. An untrained network has no
such invariance: it preserves more of the input, so matching its features
constrains more of the image.

**Encoder invariance, which is a virtue for classification, is a liability
for a kernel whose job is to say when two images should be pulled together.**

---

## 3. The caveat that must travel with this result

**This is not the paper's method.** The paper performs drifting *in* the
encoder's latent space. This harness computes the drift **in pixel space
using feature-space kernel weights** — `kernel_gradient.field` builds the
affinity matrix from the feature Gram and then applies those weights to the
*image* tensors.

Those are different algorithms. A geometry that fails as a *kernel* over
pixel space says nothing directly about the same encoder used as a *latent
space* to drift within. **Phase 17 does not refute the paper's ablation**, and
no sentence in any write-up should suggest that it does.

What it does establish is a statement about this program's own design: within
kernel drifting, a pretrained semantic encoder is a bad kernel, and an
untrained network of the same architecture is a good one.

---

## 4. What this means for the program's thesis

The encoder-independence question, as this harness poses it, now has an
answer:

> **Encoder-free drifting pays no penalty here. Raw pixels (234.72) are
> within 8.55 FID of the best geometry tested, and 138 FID better than a real
> pretrained encoder.**

And the highest-information arm delivered: **`random ≈ raw`, both far better
than `pretrained`.** Whatever benefit a deep feature map offers in this
setting comes from its *architecture*, not its *pretraining* — an untrained
network needs no pretraining data, which would make encoder-independence a
much easier target than the paper's framing implies.

The 8.55-FID edge of `random` over `raw` is real in sign across both seeds
but small, and with two seeds it rests on one degree of freedom. It should be
treated as "no worse than raw", not as a win.

---

## 5. Next

Branch (b) of the reading guide requires attacking a favourable result before
believing it. The `random_resnet` control has already dispatched the
calibration objection (§1), which leaves two:

1. **The paper's declared temperature grid on the encoder arm.** With
   normalized encoder features, τ ∈ {0.02, 0.05, 0.2} is finally meaningful —
   it collapsed the kernel on raw pixels (93.8% dead rows) because it was
   never calibrated for them. If the pretrained arm recovers at the paper's
   own operating point, §0's result is about our calibration rule rather than
   about encoders. **This is the single most important follow-up and it is
   cheap** — one geometry, three temperatures.
2. **Seeds.** Every contrast here rests on n = 2. The pretrained gap is far
   too large for that to matter; the `random − raw` edge is not, and needs
   n ≈ 6 to be worth quoting.

Both are affordable. (1) first.

---

## 6. Scope

- Not the paper's method (§3), not the paper's encoder (ResNet18/ImageNet is
  supervised, not self-supervised), not the paper's dataset or metric scale.
- FID floor here is 70.89 at 512 samples; only arms measured identically are
  comparable, and none of these numbers is comparable with published FIDs.
- 2 seeds, 30 000 steps, one bandwidth rule, one encoder input size (128px),
  one normalization (L2) — all declared in `encoders.py`, none swept.
- GPU throughout; arithmetic validated against CPU to 9.4e-7.
