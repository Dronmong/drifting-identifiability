# Encoder-Independent Kernel Drifting — why the pretrained kernel fails

## Not invariance — hypersensitivity, and it rises with depth

*Research pass on Phase 17. Code: `diagnose_phase18.py`. Artifact:
`phase18_probe.json` (+ `.sha256`). No training: feature statistics on 512
real CIFAR-32 images, GPU.*

---

## 1. The hypothesis, and its refutation

Phase 17 found a pretrained ResNet18 geometry at **FID 373.19** against raw
pixels' 234.72 and an untrained ResNet18's 226.17, with a spectral tail of
**0.4853** against real data's ~0.13 — injecting high-frequency content
rather than collapsing the cloud.

I proposed **invariance**: a pretrained encoder is trained to discard
texture, colour and position, so matching its features leaves those details
unconstrained and the field fills them with noise. The research plan's §2
anticipated the same idea.

**It is wrong, and the truth is the exact opposite.** Feature displacement
per unit of pixel displacement, L2-normalized features, same architecture and
input for both columns:

| perturbation | layer | pretrained | untrained | **ratio** |
|---|---|---:|---:|---:|
| **high-frequency noise** | layer1 | 0.0268 | 0.0062 | **4.3×** |
| | layer2 | 0.0455 | 0.0048 | **9.5×** |
| | layer3 | 0.0649 | 0.0051 | **12.6×** |
| | layer4 | 0.1118 | 0.0079 | **14.2×** |
| **blur 2×** | layer3 | 0.0653 | 0.0055 | 11.9× |
| | layer4 | 0.1182 | 0.0086 | 13.7× |
| colour shift | layer3 | 0.0545 | 0.0113 | 4.8× |
| translation 2px | layer3 | 0.0180 | 0.0051 | 3.5× |

A pretrained ResNet is **not** invariant to these distortions. It is **12–14×
more responsive** to high-frequency noise and blur than an untrained network
of identical architecture, and the gap **rises monotonically with depth** on
every perturbation.

---

## 2. The reading this supports

A kernel's job here is to say when two images should be pulled together. A
feature map that amplifies fine detail turns small high-frequency differences
into large feature distances, so the field's weights are dominated by
high-frequency structure — and the drift, a weighted combination of
image-space points, accumulates energy in exactly those directions.

That predicts the signature Phase 17 measured directly: **a spectral tail of
0.4853, 3.7× real data's**, at a healthy second moment.

The untrained network sits at ~0.005 sensitivity to everything — nearly flat,
therefore smooth, therefore a well-behaved kernel. **What makes a good kernel
here is smoothness, not semantic quality**, which is close to the opposite of
the property the paper selects an encoder for.

---

## 3. Why this reframes the program's target

If the mechanism is depth-driven sensitivity, then the useful question is not
"how do we remove the encoder" but **"how deep should the geometry be"** — and
the answer might be *shallow pretrained features*, which are still a
pretrained encoder but a far cheaper and more available one.

That would change the target from a negative (drop the encoder) to a positive
(use a shallower one), which is a more useful result and a testable one.

**Phase 18B is running that test now**: raw against pretrained layer1–4 plus
the untrained layer3 control, 5 000 steps, 2 seeds, paired within seed. The
budget is reduced deliberately — the pretrained penalty was visible at 600
steps (+38 FID) and at 30 000 (+138), so the *ordering* is measurable well
below the full budget, and ordering is what this screen needs.

Declared before the result:

- **FID rises monotonically with depth and layer1 beats raw** → depth is the
  axis, and the answer to encoder dependence is a shallower encoder.
- **FID rises with depth but no layer beats raw** → depth explains the
  failure but does not rescue the encoder; encoder-free remains the best
  option in this harness.
- **No depth ordering** → sensitivity is a correlate, not the cause, and §2's
  reading joins the list of refuted mechanisms.

---

## 4. What this does *not* say

- **Still not the paper's method.** This harness uses encoder features to
  weight a drift computed in *pixel* space; the paper drifts *within* the
  encoder's latent space. Hypersensitivity is a liability for a kernel over
  pixels and says nothing directly about a latent space one drifts inside.
- Sensitivity is measured on four hand-chosen perturbations at one input
  resolution (128px) with L2 normalization. It is a property of this
  measurement, not a general claim about ResNet features.
- No training was done in this pass; the link from sensitivity to FID is an
  inference, and 18B is what tests it.
