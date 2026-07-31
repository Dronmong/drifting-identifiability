# Encoder-Independent Kernel Drifting — Phase 16 protocol

## Can drifting produce anything a moment-matched Gaussian cannot?

*Frozen pre-outcome design. Source: `EncoderIndependentMetricAudit.md` §4.
Results go to `EncoderIndependentPhase16Results.md`.*

---

## 1. The question, and why it has a definite answer

The metric audit established that energy distance in pixel space is
saturated by matching the first two moments: a Gaussian with the data's exact
mean and covariance scores **ED² 0.1079 against a second real sample's
0.1118** — indistinguishable — while FID puts it at **240.6**, half way to
noise. R11, the program's headline reform, is a second-moment match and is
worth **6.2× on ED² and 6% on FID**.

So the program has never established that drifting produces *structure*. That
is now a single question with an unambiguous bar:

> **Does FID fall clearly below 240.6, and keep falling with budget?**

Fixed reference points, all measured at 512 samples in the audit:

| reference | FID |
|---|---:|
| real vs real — the floor | **71.7** |
| **moment-matched Gaussian — the bar** | **240.6** |
| best configuration to date | 244.0 |
| free particles | 388.1 |
| pure noise — the ceiling | 426.5 |

**FID is the primary readout.** ED², the second moment and the spectral tail
are recorded as diagnostics and must not be used to declare success. This
inverts thirteen phases of practice and is the point of the phase.

---

## 2. Design

**Stage 1 — the scaling curve.** The core question. Raw pixels (encoder-free)
with R11, width 64, field cloud 256, at budgets **600 → 3 000 → 10 000 →
30 000** steps. The 600-step point reproduces the configuration every earlier
phase used, so this stage also re-scores the historical record under FID.

**Stage 2 — the other axes, at 10 000 steps.** Whether capacity or geometry
changes the picture, run only where the curve has had room to develop:

| arm | width | cloud | geometry |
|---|---|---|---|
| S2-base | 64 | 256 | raw |
| S2-wide | 256 | 256 | raw |
| S2-cloud | 64 | 1024 | raw |
| S2-encoder | 64 | 256 | pretrained ResNet18 |

**Baselines recomputed in this run**, not quoted from the audit, so every
number on the plot comes from one seed stream: real, moment-matched Gaussian,
noise, and free particles at matched budget.

CIFAR-10 at 32×32, target ESS 0.9, Adam/2e-3, 2 fresh seeds
(`MASTER_SEED + 28000..`), GPU throughout, FID at 512 samples.

*Two seeds rather than three*: the 30 000-step arms dominate the cost and the
effect sizes at stake (240.6 versus a hoped-for large fall) are far larger
than the seed spread seen in the audit. Declared here rather than chosen
afterwards.

---

## 3. Declared outcomes

- **FID falls clearly below 240.6 and is still falling at 30 000 steps** →
  drifting produces structure; the encoder ladder becomes meaningful; Phase
  14B is worth running and the program has a real result.
- **FID plateaus at or above 240.6 for raw pixels but falls below it for the
  pretrained encoder** → this is the paper's encoder dependence, reproduced
  and quantified in our own harness. A clean negative answer to the
  program's thesis, and the most valuable outcome short of success.
- **FID plateaus near 240.6 for both** → neither the recipe nor the harness
  produces structure at this scale, no conclusion about encoders follows, and
  the honest move is to stop and write the mechanism results up for what they
  are.
- **FID rises with budget** → the recipe diverges from the data under a
  semantic metric while improving under ED². Report it; it would be the
  strongest possible statement of the audit's finding.

Each outcome closes a question. No outcome licenses another mechanism hunt.

---

## 4. Declared failure branches

- **A 30 000-step arm diverges or OOMs** → report the budget at which it
  failed with its diagnostics; do not silently drop it.
- **FID at 512 samples proves too noisy to separate arms** → re-score the
  endpoints at 2048 (the audit showed the ordering is stable there) before
  drawing any conclusion.
- **No tuning.** Budgets, widths, cloud sizes and the bar are declared above.
  The bar is 240.6 and does not move.

## 5. What Phase 16 cannot conclude

- Nothing about ImageNet or the paper's trained model; CIFAR-32 at 512-sample
  FID bounds the question and does not settle it.
- FID's floor here is 71.7, not 0, so absolute values are not comparable with
  published FIDs — only arms measured identically are comparable.
- ResNet18/ImageNet is a supervised classifier standing in for the paper's
  self-supervised encoder.
- The anchor stays disabled; the geometry thread stays closed.
