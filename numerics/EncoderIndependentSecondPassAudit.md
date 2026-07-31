# Encoder-Independent Kernel Drifting — second research pass

*Implementation audit, regime finding, and the design of the next
experimental phase. Follows `EncoderIndependentPhase1Diagnosis.md` and the
seven reforms in `EncoderIndependentReformedScreenProtocol.md`. Code:
`numerics/encoder_independent_drifting/audit_phase2.py`. Artifact:
`phase2_cifar.json` (+ `.sha256`). Nothing here feeds a gate.*

---

## 0. Executive summary

Five audits (A1–A5) plus a code review. Two produced new reforms, one closed
a standing question with a clean negative, and the last two together
**found the regime this program has been missing since it started**.

| # | Question | Answer |
|---|---|---|
| **A1** | Is the metric's target level reachable by this generator class? | **No.** A fresh real sample scored 1.13–2.78, not 1.0 → **reform R8** |
| **A2** | Does the unmasked self-term bias the field? | **No.** Ratio 0.978–1.002, cosine ≈1.000. Question closed |
| **A3** | What does the R3 projection cost? | 10 ms/call of redundant SVD → **reform R9**, now free |
| **A4** | Can a *synthetic* testbed make pixel geometry fail? | **No.** Pixel NCR never exceeds ~1.0 across resolution 16–32, translation ±4 |
| **A5** | Does pixel geometry fail on *real* images, and does fixed geometry fix it? | **Yes, and yes.** On CIFAR-10 pixel k-NN content accuracy is 0.267; fixed wavelet reaches **0.390** (+46%), matching a small supervised encoder, while a learned autoencoder reaches only 0.282 |

A4 and A5 together are the finding of this pass. Every previous phase ran on
synthetic targets where raw pixel distance already worked, which is why fixed
geometry could never demonstrate value. On real images the situation inverts:
pixel geometry is genuinely weak, fixed compositional geometry recovers most
of the gap, and — the surprise — an *unsupervised learned* encoder at this
data scale recovers almost none of it.

**CIFAR-10 at 16×16 is also admissible at 300 steps** (skyline precision
1.000 against a 0.482 bar), a gate four of nine synthetic targets failed.
For the first time the program has a testbed that is both solvable and
discriminating, and the next phase is designed on it.

---

## 1. A1 — the metric's target level is not reachable

| target | fresh real sample | noiseless prototypes | capacity ceiling |
|---|---:|---:|---:|
| checkerboard | **2.777** | 4.134 | 3.787 |
| texture_blocks | **1.454** | 2.000 | 2.004 |
| rings_islands | **1.541** | 3.062 | 1.974 |
| pinwheel | **1.131** | 2.271 | 1.653 |

*`fresh real` — an independent target sample, nominally 1.0. `noiseless` —
exact prototypes at the true mixture weights. `capacity ceiling` — the
generator fitted directly to 512 target images by regression.*

1. **A fresh real sample does not score 1.0**, it scores 1.13–2.78. The
   denominator was a single realization of energy distance between two
   samples of the *same* law — a near-zero quantity with large relative
   fluctuation. Absolute scores in every earlier document are meaningful only
   well above ~3.
2. **The capacity ceiling is 1.65–3.79.** The generator maps a 32-dimensional
   latent through a deterministic network, so its output lies on a manifold of
   dimension ≤ 32 while the synthetic targets have full 768-dimensional
   support. Phase-1's best arm scored 8.37, so the real headroom was ~4×, not
   the ~8× a 1.0 reference implied.
3. **`texture_blocks` is unusable**: its capacity ceiling has precision
   **0.000**. A generator fitted directly to 512 of its images still lands
   outside the calibrated support.

### Reform R8 — average the null

`null_reference` now takes the median over `NULL_REPEATS = 5` independent
draws and reports `null_spread`:

| target | before | after |
|---|---:|---:|
| checkerboard | 2.777 | **1.676** |
| texture_blocks | 1.454 | **1.283** |
| pinwheel | 1.131 | **1.160** |

The floor is tightened and now *reported* rather than assumed. Honest
statement: **the achievable floor is ≈1.2–1.7 and differences below it are
not measurable.**

---

## 2. A2 — the self-term does not bias the field *(clean negative)*

The generated cloud is reused as negatives without the paper's eye mask, so
`k(x_i, x_i) = 1` enters the negative denominator without contributing to the
numerator.

| target | family | direction | ‖unmasked‖/‖masked‖ | cosine |
|---|---|---|---:|---:|
| checkerboard | raw | kernel gradient | 0.986 | 1.000 |
| checkerboard | wavelet | standard | 1.002 | 0.979 |
| texture_blocks | raw | kernel gradient | 0.978 | 1.000 |
| texture_blocks | wavelet | kernel gradient | 0.988 | 1.000 |

≤2.2% in magnitude, undetectable in direction. `self_mask = False` is safe at
batch 64, consistent with the repository's Phase-C finding. **No reform
needed** — recorded so the question is not reopened.

---

## 3. A3 — the projection was repeating the same SVD

| family | direction | ms/call |
|---|---|---:|
| raw | kernel gradient | 1.83 |
| raw | **projected** kernel gradient | **11.44** |
| wavelet | kernel gradient | 25.78 |
| wavelet | **projected** kernel gradient | **35.51** |

Every branch in an arm sees the same probes and anchors, so a four-branch arm
computed the identical factorization four times.

### Reform R9 — factorize once per step

`data_span_basis` / `apply_data_span_projection` split the factorization from
its application; `_branch_drifts` computes it once. Measured on a full
training step:

| direction | ms/step (4 branches) |
|---|---:|
| kernel gradient | 144.3 |
| projected kernel gradient | **139.2** |

The projection is now free. `field_with_snr` explicitly clears the cache
because its half-batches have different anchors — a shared basis there would
project onto the wrong subspace. Two regression tests cover both.

---

## 4. A4 — the synthetic route to a discriminating testbed is closed

**Nuisance confusion ratio (NCR)** = median distance between two samples of
the *same content* under *different nuisance*, over the median distance
between *different content* under the *same nuisance*. `NCR < 1` means the
geometry groups by content; `NCR > 1` means it groups by nuisance and is
actively misleading.

Four content prototypes under translation and a smooth warp:

| resolution | shift | warp | **pixel** | wavelet | scattering | randconv |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 0.0 | 0.897 | 0.207 | 0.313 | 0.575 |
| 16 | 4 | 0.0 | **1.016** | 0.317 | 0.426 | 0.726 |
| 24 | 2 | 0.0 | 0.778 | 0.238 | 0.298 | 0.520 |
| 24 | 4 | 0.3 | **1.021** | 0.373 | 0.427 | 0.647 |
| 32 | 2 | 0.0 | 0.639 | 0.282 | 0.332 | 0.456 |
| 32 | 4 | 0.3 | 0.970 | 0.425 | 0.451 | 0.662 |

Pixel NCR tops out at 1.02 — it reaches indifference but never becomes
misleading. Fixed geometry is consistently 2–3× better, but **better is not
necessary**: a kernel already grouping by content is not the bottleneck.
Reaching NCR > 1 would need translations comparable to the image size, at
which point "same content" stops being meaningful.

Reform R7 asked for a harder synthetic testbed. **It cannot be built this
way**, which is what sent this pass to real data.

---

## 5. A5 — on real images, the premise holds and fixed geometry delivers

CIFAR-10, downsampled, 2048 images, class label as content. Class labels are
an ORACLE diagnostic for measurement only and enter no objective.

### The statistic had to be replaced

NCR was run first and **saturates on real data**:

| geometry | NCR (16×16) |
|---|---:|
| pixel | 0.956 |
| wavelet | 0.941 |
| scattering | 0.952 |
| randconv | 0.923 |
| learned unsupervised | — |
| **learned supervised (100% train acc)** | **0.922** |

The entire representational range — from raw pixels to a supervised encoder
that has memorized the labels — spans 3.4%. A ratio of median distances
cannot discriminate representations in high dimension, because in high
dimension the median within-class and between-class distances are genuinely
close even for good representations. **Without the supervised control this
would have been read as "fixed geometry fails on real images", which is
false.**

The replacement is **leave-one-out k-NN content accuracy** (k = 10), which
reads the local neighbourhood a kernel actually uses. Chance = 0.100.

### The result

| geometry | k-NN 16×16 | k-NN 24×24 | vs pixel |
|---|---:|---:|---:|
| **pixel** | **0.267** | **0.260** | — |
| wavelet | **0.390** | 0.354 | **+46% / +36%** |
| scattering | 0.386 | **0.389** | +45% / +50% |
| random conv | 0.287 | 0.294 | +7% / +13% |
| *control:* learned unsupervised | 0.282 | 0.272 | +6% / +5% |
| *control:* learned supervised | 0.406 | 0.389 | +52% / +50% |

Four readings, in order of importance:

1. **Pixel geometry is genuinely weak on real images** — 0.267 against a
   chance level of 0.100. This is the regime the plan is premised on and the
   one the synthetic suite never reached.
2. **Fixed wavelet and scattering recover most of the gap**, +36–50% relative
   over pixels, and they do it with no training and no data.
3. **They match the supervised control** (0.386–0.390 against 0.389–0.406).
   That control is a small CNN trained on 2048 images to 100% training
   accuracy, so it is a *weak* ceiling, not a pretrained encoder — but at this
   data scale, fixed compositional features are competitive with learning.
4. **The unsupervised learned encoder recovers almost nothing** (0.282 vs
   0.267). This is the most surprising number in the pass: at this scale, a
   denoising autoencoder is barely better than raw pixels, while a fixed
   scattering transform is 46% better. It is a direct argument for fixed
   compositional geometry over cheap representation learning in the
   small-data regime.

Kernel health on CIFAR is not a differentiator: raw ESS 0.799, wavelet ESS
0.800 at 16×16. The advantage is in *what the kernel ranks*, not in how
peaked it is.

### The testbed is also solvable

| budget | skyline precision | required | score | admissible |
|---:|---:|---:|---:|---|
| 300 steps | **1.000** | 0.482 | 2.120 | **yes** |
| 1200 steps | 1.000 | 0.482 | 1.474 | yes |

CIFAR-16 clears the R1 admissibility bar at the smallest budget tested, with
a skyline score of 2.12 against a metric floor of ≈1.2–1.7. Four of the nine
synthetic targets never cleared it at any budget.

---

## 6. Consolidated implementation state

**Nine reforms implemented and unit-tested; 90 tests, all passing.**

| Reform | Source | Status |
|---|---|---|
| R1 skyline arm, admissibility, oracle budget | Phase-1 diagnosis | 4 tests |
| R2 honest unnormalized loss | Phase-1 diagnosis | 3 tests |
| R3 projected kernel gradient | Phase-1 diagnosis | 4 tests |
| R4 composite score v2 | Phase-1 diagnosis | 5 tests |
| R5 zero-set gate (G0.5) | Phase-1 diagnosis | gate passes |
| R6 coarse-to-fine anchor schedule | Phase-1 diagnosis | 5 tests |
| R7 rebuild the target suite | Phase-1 diagnosis | **resolved by A5** — CIFAR-16 |
| **R8 averaged null reference** | **A1** | 2 tests |
| **R9 shared projection factorization** | **A3** | 2 tests |

Also added, from A5's methodology failure:

**R10 — a representational statistic must be validated against a learned
control before it is trusted.** NCR would have produced a confident false
negative on real data. Any future "does this geometry carry content" claim
must report a learned-encoder control that establishes the statistic's
dynamic range, and prefer k-NN accuracy over distance ratios in high
dimension.

---

## 7. The next experimental phase

### 7.1 Phase 2A — entry conditions on CIFAR-16 *(partly verified)*

| condition | status |
|---|---|
| skyline admissibility, budget derivation | **verified**: admissible at 300 steps |
| pixel geometry is the bottleneck | **verified**: k-NN 0.267 vs 0.390 for wavelet |
| G0.5 zero-set reachability per (geometry, direction) | **to run on CIFAR** |
| kernel health / collapsed-row fraction | measured: ESS ≈0.80 both, no differentiator |

Only G0.5 remains. It must be run on CIFAR before any arm, and it decides
which direction rule each family may use — on synthetic targets it rejected
`wavelet::kernel_gradient` (2.39 against a 2.00 threshold) while admitting
`wavelet::standard` (0.68).

### 7.2 Phase 2B — the decisive comparison

Four arms and a skyline, not Phase-1's nine:

| ID | Anchor | Geometry | Direction | Purpose |
|---|---|---|---|---|
| B0 | — | raw pixel | standard | the baseline that won Phase 1 |
| B1 | — | wavelet | standard | the +46% k-NN candidate |
| B2 | — | scattering | standard | the other +45–50% candidate |
| B3 | scheduled | wavelet | standard | does the anchor still help? |
| SKY | — | — | sliced Wasserstein | skyline; never in a gate |

Every arm uses standard displacement. G0.5 rejects the kernel-gradient rule
for structured kernels, Phase 1 measured it 1.86× worse, and R3's projection
does not repair it (2.21 vs 2.39). **The plan's section 6.3 hypothesis is
abandoned, not re-tested.**

Design: CIFAR-10 at 16×16, budget from the skyline (300 steps clears; run at
600 for headroom), 3 seeds, v2 composite with per-component verdicts, all
nine reforms active.

**Gate.** B1 or B2 beats B0 by ≥10% on the v2 composite with a bootstrap
upper bound below 1, **and** on a majority of components individually, **and**
on every seed — with B3 reported for the anchor's contribution and SKY
reported as the ceiling.

### 7.3 Why this is the right experiment now

It is the first time all three preconditions hold simultaneously:

- the **testbed is solvable** (skyline precision 1.000 at 300 steps) — Phase 1
  failed this and its result was uninterpretable;
- the **geometry question is live** (pixel k-NN 0.267 vs wavelet 0.390) —
  Phase 1 failed this and fixed geometry had nothing to win;
- the **instrumentation is sound** (nine reforms, 90 tests, honest loss,
  validated metric, zero-set pre-screen).

The expected outcomes are all informative. If B1/B2 beat B0, the program has
its first positive result, on real images, with no pretrained encoder
anywhere. If they do not, then a 46% better neighbour ranking does *not*
translate into better drifting — which, given G0.5 and the Phase-1 diagnosis,
would point squarely at the objective rather than the geometry, and would be
worth knowing.

### 7.4 Standing alternatives

If Phase 2B is negative, two threads remain better than persisting:

**(a) The anchor as the subject.** It passed every condition asked of it,
costs almost nothing, and connects directly to
`DriftingIdentifiability/FeatureSpaceIdentifiability.lean`. "Can a spectral
anchor make an arbitrary drifting method verifiably source-correct at
negligible cost?" is a correctness contribution independent of geometry.

**(b) Zero-set reachability as the subject.** G0.5 predicted the whole
Phase-1 ranking from a twenty-second particle run. "Zero-set reachability of
a kernel field predicts generative performance" is a diagnostic contribution
testable across every kernel family already implemented.

---

## 8. What this pass does not establish

- No arm was trained on CIFAR; A5 is a property of *representations*, not of
  generated samples. A better neighbour ranking is necessary, not sufficient —
  Phase 1 showed the zero-set matters independently.
- The supervised control is a small CNN on 2048 images, not a pretrained
  encoder. It bounds the statistic's range; it does not bound what a real
  encoder achieves, which is substantially higher.
- A4's sweep uses one synthetic content family; "pixel geometry does not fail"
  is a statement about that family at those resolutions.
- A1's capacity ceiling is measured on four targets at one generator size.
- R8's averaged null does not reach 1.0; the residual floor ≈1.2–1.7 is a
  property of the metric and sample size.
- Nothing here concerns ImageNet, FID, or the paper's trained model.
