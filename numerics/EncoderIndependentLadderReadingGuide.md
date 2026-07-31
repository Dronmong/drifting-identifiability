# Encoder-Independent Kernel Drifting — how to read the Phase-17 ladder

## Power, proxies, and the decision tree, settled before the result lands

*Analysis written while Phase 17 runs, from sealed artifacts only. No new
runs; no GPU. The point is to fix in advance what each possible outcome
licenses, so the reading is not chosen after seeing the numbers.*

---

## 1. No cheap diagnostic predicts FID

Pooling every generated-sample measurement this program has recorded with
both metrics (n = 30, Phases 14A–17):

| relationship | Spearman |
|---|---:|
| FID vs **ED²** | **+0.302** |
| FID vs spectral tail | +0.138 |
| FID vs second moment | −0.070 |

**None of the program's diagnostics is a usable proxy for FID.** ED² is the
best of them at ρ = 0.30, which is directionally right and practically
useless; the spectral tail and the second moment — the two quantities the
mechanism work identified as controlling — carry essentially no information
about semantic quality.

*Consequence:* every future comparison must measure FID directly. There is no
shortcut, and any argument of the form "the tail improved, so the samples
improved" is unsupported.

---

## 2. What the running ladder can and cannot resolve

Seed spread at 30 000 steps, pooled across Phase 16 and Phase 17 (raw + R11,
the only configuration measured twice at that budget):

> 235.86, 228.42, 219.62, 249.82 → **mean 233.4, sd 12.8**

Minimum gap resolvable at 2 standard errors of an unpaired difference:

| seeds per arm | resolvable gap |
|---:|---:|
| **2** *(the running design)* | **25.6 FID** |
| 3 | 20.9 |
| 4 | 18.1 |
| 6 | 14.8 |
| 8 | 12.8 |

For reference, the geometry spread measured at 600 steps was **33.3 FID**
(raw 261.9 → pretrained 295.2).

**So two seeds is adequate only if the geometry gap at 30 000 steps is as
large as it was at 600.** Every arm improves with budget, and gaps commonly
shrink as models get better, so this is the live risk in the running
experiment — flagged before the result, not after.

---

## 3. The design is already paired — analyse it that way

`run_phase17` derives the generator initialization and the latent stream from
the *seed*, not from the geometry, so every geometry sees the same
initialization and the same latents. That is a paired design and it should be
analysed pairwise within seed rather than by comparing medians.

Verified on Phase 14A's 5 × 3 grid:

| contrast | paired mean | SE | t |
|---|---:|---:|---:|
| pretrained − raw | **+33.32** | 9.29 | **+3.59** |
| random − raw | **+21.32** | **2.76** | **+7.72** |
| pretrained − random | +12.00 | 6.85 | +1.75 |

The variance decomposition on the same grid: between-geometry sd **16.20**,
between-seed sd **5.84** (shared, removable by pairing), residual sd **6.34**.

The `random − raw` contrast is the clearest illustration: paired SE 2.76
against a per-arm sd of ~7.5, because both arms move together across seeds.

**Instruction for reading Phase 17:** report the three pairwise within-seed
differences with their standard errors, not just the ranking of medians. With
2 seeds the SE has 1 degree of freedom and is weak, so it should be quoted as
a range, not a p-value.

---

## 4. The decision tree, fixed in advance

Let **Δ = FID(pretrained) − FID(raw)**, paired within seed. The bar is the
moment-matched Gaussian at 248.08; the floor is 70.89.

### (a) Δ < −26 — the pretrained encoder clearly wins

The geometry the paper depends on does real, measurable work in our harness.
**The thesis takes a hit**: encoder-free drifting pays a quantified penalty.

*Route:* the question becomes whether anything closes the gap. Two candidates,
in order — **turn the anchor back on** (disabled since Phase 2; it is the only
mechanism addressing source correctness, and the one part of the original
plan never tested at a meaningful budget), and re-open Branch B *only* with
the encoder gap as a target to hit rather than as a blind search.

### (b) Δ > +26 — raw pixels clearly win

A real pretrained encoder is *worse* than no encoder at this scale. That
inverts the paper's ablation and is the program's strongest possible result —
but it is also the outcome most likely to be an artifact, so it must be
attacked before it is believed:

*Route:* (i) check the encoder arm is not simply mis-calibrated — sweep its
bandwidth around the ESS-0.9 point, since a feature space with different
distance statistics may need a different operating point even at matched ESS;
(ii) run the paper's own declared temperature grid on the encoder arm, which
is now finally meaningful with normalized encoder features; (iii) only then
report it, scoped to CIFAR-32 and 512-sample FID.

### (c) |Δ| < 26 — no resolvable difference

The likeliest outcome given §2, and it is **not** a null result — it is an
underpowered one, and must be labelled that way rather than reported as
"encoder-free matches encoder-based".

*Route:* extend **raw and pretrained only** to 6 seeds (dropping `random`,
which §3 shows is the best-resolved contrast and therefore the least urgent).
That buys a resolvable gap of ~15 FID at roughly 4 more GPU hours, and it is
the cheapest path to a defensible number.

### In every branch

The `random_resnet` arm answers a question the paper's ablation does not: if
`pretrained ≈ random`, then whatever the encoder contributes is the
*architecture's* feature map, not its pretraining — which would make
"encoder-independence" a much easier target than the paper implies, since an
untrained network needs no pretraining data. Phase 14A's paired estimate
(+12.00 ± 6.85, t = 1.75) is suggestive but not resolved.

---

## 5. Caveats on this analysis

- The 30 000-step sd rests on **four** measurements. It is the best estimate
  available and it is thin; the resolvable-gap table should be read as an
  order of magnitude, not a precise power calculation.
- The variance decomposition and the paired SEs come from Phase 14A at **600
  steps**. Whether the seed/geometry variance ratio holds at 30 000 is
  assumed, not measured.
- All of it is CIFAR-32 with 512-sample FID, floor 70.89.
