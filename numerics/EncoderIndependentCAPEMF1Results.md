# CAP-EMF-1 results — a one-call encoder-free generator that works and is bad

**Date:** 2026-08-06
**Protocol:** [`EncoderIndependentCAPEMF1Protocol.md`](EncoderIndependentCAPEMF1Protocol.md)
**Status:** budget stop at 650 000 of a declared 750 000. Developmental, one unit.

*Artifacts: `stage_cap/cap_emf1_unit.json` (sha256 `453554bde2dead45…`),
`stage_cap/cap_evaluation.json` (`92346a212c03054c…`), preflight
`cap_preflight_gpu.json`, frozen trunk `cap_emf1_step650000_ema.pt`
(`b55b2a62bfc44e54…`), grids under `stage_cap/grids/`. All hashes verified
identical between the rented GPU and local disk.*

---

## 1. The one-sentence result

> A 37.7 M-parameter raw-pixel Euler Mean Flow model, trained on unconditional
> CIFAR-10 with **no encoder, no VAE, no teacher and no perceptual loss**,
> produces recognizable objects and coherent scene layout in **exactly one
> network call** — at **FID-50k 112.9**, which is very poor, and while
> **diverging in the high-frequency bands** over the second half of training.

Both halves matter. It generates; it generates badly; and the way it fails is
diagnosable.

---

## 2. What ran

| item | value |
|---|---|
| data | unconditional CIFAR-10, 50 000 train images |
| model | U-ViT, patch 2 → 256 tokens, width 384, depth 12, AdaLN-Zero, conv refiner |
| parameters | **37 726 863** (DDPM is 35.7 M) |
| objective | direct-`x` Euler Mean Flow, JVP-free, Equation 18 |
| inference | **one network call**, asserted by hook |
| optimizer | AdamW `1e-4`, betas (0.9, 0.95), 5 000-update warmup |
| batch | 64 (single microbatch, accumulation 1) |
| precision | TF32 matmul, FP32 storage and accumulation |
| EMA | 0.9999 |
| hardware | rented RTX 4090, torch 2.8.0+cu128 |
| **updates** | **650 000** of 750 000 declared (86.7%) |
| **images seen** | **41 600 000** = 832 epochs |
| wall time | 148 804 s = **41.33 h** training |
| rate | **0.2289 s/update** |
| model forwards | 82 526 898 |
| non-finite updates | **0** |
| clipped updates | 99 595 = **15.3%** |

---

## 3. Primary results — sealed evaluation, run once

The 10 000 test images were opened exactly once, after the checkpoint was
frozen and hashed. The evaluator emitted its non-primary-checkpoint warning as
designed.

| metric | value | reference |
|---|---:|---|
| **FID-50k vs train** | **112.94** | DDPM 3.17, EDM 1.97 |
| FID vs sealed test | 113.90 | held-out cross-check |
| KID | 0.10106 | unbiased at this size |
| precision | 0.372 | fraction of samples inside the real manifold |
| recall | 0.2412 | fraction of real data covered |
| F1 | 0.2926 | |
| density | 0.2503 | |
| coverage | 0.1686 | |

FID follows the standard CIFAR-10 protocol — 50 000 generated against the
50 000 training images — so the number is directly comparable to published
work. **It is roughly 35× worse than a well-trained DDPM.**

The two FIDs agreeing to within 1.0 (112.94 train, 113.90 test) says the model
is not overfitting the training split.

---

## 4. Distributional diagnostics — where it fails

This is the informative part.

### 4.1 Effective rank

| | value |
|---|---:|
| generated | **77.04** |
| real CIFAR-10 | **9.13** |
| ratio | **8.44** |

Real natural images have a highly concentrated covariance spectrum — a
participation ratio of ~9 across 3 072 pixel dimensions. The generated cloud
spans **77** effective directions. The samples are far more *spread* in pixel
space than real images: a noise-like, flat spectrum rather than the sharply
decaying one natural images have.

### 4.2 Multiscale energy

| Haar band | ratio to real |
|---|---:|
| LL (low frequency) | **0.931** |
| LH | 3.846 |
| HL | 3.497 |
| **HH (diagonal high)** | **6.371** |

**Low-frequency structure is nearly correct.** Every detail band is 3.5–6.4×
in excess. The model has learned where things are and what colour they are, and
paints far too much fine texture on top.

### 4.3 Radial spectrum slope

| | alpha |
|---|---:|
| real | 3.596 |
| generated | 2.913 |
| gap | **−0.683** |

The generated power spectrum decays more slowly than natural images. Same
finding from an independent estimator: excess high-frequency content.

---

## 5. Memorization — clean

| statistic | value |
|---|---:|
| nearest-train distance, median | 27.12 |
| real-to-real distance, median | 36.80 |
| **ratio** | **0.737** |
| nearest-train distance, 5th pct | 20.50 |
| nearest-train distance, minimum | 6.61 |
| duplicate rate | **0.0000** |
| distinct nearest-train images | 0.569 |

A memorizing model would sit near zero on the ratio. At **0.737** the samples
are almost as far from their nearest training image as two random training
images are from each other. **The coverage is genuine, not recall of the
training set** — which is what makes the low recall of 0.241 a real (if poor)
measurement rather than an artifact.

---

## 6. The trajectory — the run diverged

EMA checkpoints, train-only diagnostics:

| checkpoint | moment | LL | HH | visual |
|---:|---:|---:|---:|---|
| 250 000 | 0.848 | 0.769 | 2.204 | painterly, clear objects |
| 300 000 | 0.862 | 0.790 | 2.054 | ~unchanged |
| 500 000 | 1.064 | 0.945 | 3.191 | sharper and crunchier |
| **650 000** | **1.161** | **0.971** | **5.898** | **fragmented** |

Low-frequency agreement improved monotonically, 0.769 → 0.971. High-frequency
excess grew monotonically, 2.05 → 5.90. **Visually, 500 000 is clearly better
than 650 000**; the final 150 000 updates made the samples worse.

The budget-stop rule fixed the result at the last completed checkpoint. That is
why 650 000 is reported rather than the better-looking 500 000: choosing on
appearance after the fact is precisely the post-hoc selection the protocol
forbids. **The rule cost us the prettier number and it was right to.**

---

## 7. The capability gate says PASS, and it means almost nothing

```
train_only_gate: PASS   failed=[]
H1_second_moment  True    H5_haar_detail   True
H2_centered_var   True    H6_finite        True
H3_rank_noncollapse True  H7_clip_fraction True
H4_haar_hh        True    H8_one_call      True
```

This is a **PASS on a model with 6.4× the target diagonal high-frequency energy
and 8.4× the effective rank.** Two independent reasons it carries no
information:

1. **Every threshold is a floor.** H4 requires HH ≥ 0.50 because the predecessor
   run's failure was too *little* detail (S3R reached 0.159). Nothing in the
   gate can object to too much. H3's rank rule caps `best` at 1, so
   over-dispersion is invisible to it by construction.
2. **H7 passed on a fabricated value.** The recovery file does not carry the
   windowed clip counters, so `finalize.py` substitutes `0.0`, and
   `0.0 < 0.05` is trivially true. **The real run-wide clip rate is 15.3%** —
   three times the threshold H7 exists to enforce.

This was predicted before the run and demonstrated in rehearsal at step
450 000. The grids and FID are the evidence; the gate is not.

---

## 8. Secondary and exploratory — post-hoc EMA

Averaging the trailing 8 parameter snapshots gives a **200 000-update window**
against the declared EMA's ~10 000:

| | FID-50k vs train |
|---|---:|
| primary, declared 0.9999 EMA at 650 000 | 112.94 |
| **post-hoc EMA, 200 k window** | **83.65** |

A **26% improvement**, and visibly cleaner samples. Computed against the
training set, so the sealed split was not reopened.

Labelled secondary and exploratory as declared before the run. It does not
replace the primary result. That the remedy for high-frequency divergence is
"average over a much longer window" is consistent with the diagnosis in §9.

---

## 9. Corrected diagnosis: sparse joint inference coverage and coefficient stress

> **Post-run correction.** The table below is one logged 64-row batch, not a
> run-wide occupancy table. The aggregate of all 1,300 logged batches is given
> in `EncoderIndependentCAPEMF1ExtendedAnalysis.md`: about 0.38% of rows had
> `t > .95`, consistent with the declared logit-normal law. The exact
> `(t,h)=(1,1)` inference point remains measure-zero and its *joint
> neighborhood* is extremely sparse, but “never trained” was too strong.

The time-bucket diagnostic, added shortly before launch specifically to answer
this question:

| t bucket | share of batch | mean raw MSE |
|---|---:|---:|
| 0.3–0.6 | 0.297 | 0.121 |
| 0.6–0.8 | 0.484 | 0.521 |
| **0.8–0.9** | 0.203 | **15.00** |
| 0.9–0.95 | 0.016 | 0.256 |
| **0.95–1.0** | **0.000** | — |

Two findings:

**The one-call sampler evaluates at `(t,h)=(1,1)`, while the
logit-normal(0.8, 0.8) law gives its neighborhood very little joint mass.**
The endpoint bucket is sparse rather than empty, and the exact point still
requires extrapolation from continuous-time samples.

**A fifth of every batch sits in an ill-conditioned band.** Equation 18's
coefficient is `(t − r − δ)·t / max(r, 0.02)`, which reaches ≈ 35 when `t ≈ 0.85`
and `r` sits at its 0.01 floor. Those rows carry enormous targets — 30–120× the
error of their neighbours. The adaptive weighting normalises them in the loss,
which is why training never destabilised (zero non-finite updates in 650 000).
With adaptive power one, large residuals do **not** automatically dominate the
output-gradient norm; the historical logs did not measure per-row parameter
gradients. The 15.3% clipping rate is evidence of global optimization stress,
not proof that these particular rows caused it.

Ill-conditioned targets together with sparse inference-corner coverage remain
a coherent mechanism for high-frequency divergence. They are hypotheses, not
an identified root cause; CAP2 therefore logs aggregate occupancy and sampled
actual parameter-gradient contributions and uses a controlled sampler screen.

---

## 10. Budget stop, recorded

| | |
|---|---|
| declared horizon | 750 000 updates / 48 M images |
| reached | **650 000 / 41.6 M (86.7%)** |
| reason | budget exhausted |
| rule | last completed checkpoint is the result; no checkpoint chosen on its numbers |
| GPU cost | ≈ $32 of a $35 balance, ~42.5 h wall |

The stop was executed by a watchdog that waited for the 650 000 EMA checkpoint,
confirmed the file had finished writing, waited a further two minutes, and
stopped the run cleanly. Training and evaluation together consumed the balance
with roughly $2.50 to spare.

---

## 11. What this licenses

**May claim:**

> On unconditional CIFAR-10 at 32×32, a raw-pixel one-call Euler Mean Flow
> generator with no encoder, VAE, teacher or perceptual loss produces
> recognizable objects and coherent scene composition, at FID-50k 112.9 under
> the standard protocol, after 41.6 M images — and exhibits monotone
> high-frequency divergence over the second half of training.

**May not claim:**

- that this is a good generator — FID 112.9 is ~35× published CIFAR-10 results;
- that it is a *capable* foundation in the sense ASFD's precondition P1
  intended — see §12;
- anything about ImageNet, higher resolution, or class-conditional generation;
- replication — one training unit, one seed;
- that the divergence mechanism in §9 is established rather than hypothesised;
- that the `PASS` gate verdict is evidence of anything (§7).

---

## 12. Consequence for ASFD

ASFD's precondition **P1** requires a foundation that passes a capability gate
including *"visibly recognizable uncurated automobiles"* and adequate
high-frequency retention. On the letter of the declared thresholds this run
passes. On the substance it does not: the trunk that ASFD would freeze and use
as a **feature geometry** is a model whose output carries 6.4× excess diagonal
high-frequency energy and 8.4× excess effective rank.

ASFD's feature qualification gate G7 demands per-band sensitivity
`ρ_b ∈ [0.25, 4.0]` in **every** Haar band. A trunk trained to this endpoint is
a poor prospect for it.

**Recommendation: do not fork ASFD from this checkpoint.** Fix the time
sampling (§9), re-run the foundation, and re-assess. The cost is comparable to
this run and it addresses a diagnosed cause rather than building a correction
on a diverged trunk.

Full process analysis: [`EncoderIndependentCAPEMF1Retrospective.md`](EncoderIndependentCAPEMF1Retrospective.md).
