# Encoder-Independent Kernel Drifting — Phase 16 results

## Drifting does produce structure — and ED² moves the opposite way

*Protocol: `EncoderIndependentPhase16Protocol.md`, frozen before the run.
Code: `run_phase16.py`. Stage 1 complete; Stage 2 partial (see §4).
CIFAR-10 at 32×32, 2 seeds, GPU, FID at 512 samples, R11 on every arm.*

---

## 0. The answer

Baselines recomputed in this run, one seed stream:

| reference | FID |
|---|---:|
| real vs real — floor | **71.31** |
| **moment-matched Gaussian — THE BAR** | **254.77** |
| pure noise — ceiling | 430.35 |

**Stage 1, the scaling curve** (raw pixels, encoder-free, R11, width 64,
cloud 256):

| steps | seed 0 | seed 1 | **median** | vs bar |
|---:|---:|---:|---:|---|
| 600 | 286.63 | 298.04 | **292.34** | above |
| 3 000 | 283.98 | 291.99 | **287.99** | above |
| 10 000 | 248.98 | 269.02 | **259.00** | above |
| **30 000** | **235.86** | **228.42** | **232.14** | **BEATS** |

> **FID falls monotonically with budget — 292 → 288 → 259 → 232 — and at
> 30 000 steps both seeds beat the moment-matched Gaussian.** The curve is
> still falling at the largest budget tested.

This is the protocol's **first declared outcome**: encoder-free drifting on
raw pixels produces structure that a Gaussian matching the data's mean and
covariance does not. It is the first result in this program that survives a
semantic metric.

The margin is modest — 232.14 against a bar of 254.77, roughly 9% — and the
floor is 71.31, so the samples remain far from real data. What the phase
establishes is *direction and existence*, not quality.

---

## 1. ED² moves the other way

The same runs, scored both ways:

| steps | FID (median) | ED² (median) |
|---:|---:|---:|
| 600 | 292.34 | **0.339** |
| 3 000 | 287.99 | 0.353 |
| 10 000 | 259.00 | 0.366 |
| 30 000 | **232.14** | **0.348** |

**FID improves 26% across the sweep while ED² is flat-to-worse.** At the
single best FID run (seed 0, 30 000 steps, 235.86) ED² is **0.5051** — the
*worst* ED² of any Stage-1 arm, and worse than the same configuration at 600
steps (0.3363).

Had this phase been scored the way the previous thirteen were, it would have
concluded that training longer makes the model worse. The metric audit
predicted exactly this, and here it is on a single set of runs: **the two
metrics disagree on the sign of the effect of training budget.**

The second moment reaches 0.999 and 0.937 at 30 000 steps, and the spectral
tail sits at 0.134–0.148 against real data's ~0.13 — so by the program's own
diagnostics these arms are *correct*, while FID says they are 3.3× the floor.
That gap is the honest summary of what thirteen phases of second-moment work
bought.

---

## 2. What this changes

**The thesis is testable again.** Phase 14A could not interpret the encoder
ladder because every arm sat where a moment-matched Gaussian sits. At 30 000
steps the encoder-free arm is clearly below that. A ladder run at this budget
would be measuring geometry rather than moment-matching.

**But the budget is the binding constraint, not the geometry.** The
difference between the historical configuration (600 steps, FID 292) and the
same recipe trained 50× longer (232) is far larger than every geometry
difference measured in 14A (255–302). **The program spent thirteen phases
varying things that matter less than the training budget it never varied.**

---

## 3. Recommended next step

**Re-run the encoder ladder at 30 000 steps.** That is Phase 14A's design at
the budget where FID discriminates: raw pixels against pretrained ResNet18,
random ResNet18 and the degraded control, FID primary. It is the experiment
14A was meant to be, and it is now interpretable.

Cost is the obstacle: 30 000 steps is ~35 min per run on this card, so a
five-geometry ladder at two seeds is ~6 hours, and the encoder arms are
slower still. Worth scoping to three geometries — pretrained, random, raw —
which is the comparison that carries the thesis.

---

## 4. Stage 2, and an honest incompleteness

Stage 2 asked whether capacity, cloud size or geometry change the picture at
10 000 steps. **Only `S2_base` completed:**

| arm | width | cloud | geometry | FID | ED² |
|---|---:|---:|---|---:|---:|
| S2_base | 64 | 256 | raw | 248.89 / 265.63 | 0.364 / 0.283 |
| S2_wide | 256 | 256 | raw | **did not complete** | — |
| S2_cloud | 64 | 1024 | raw | not reached | — |
| S2_encoder | 64 | 256 | pretrained | not reached | — |

`S2_wide` ran for over 30 minutes on a single seed without finishing, at
5844 MiB of the card's 6141 MiB. At 16× the convolutional cost of width 64
and near the memory limit, the width-256 arm is **impractical on this GPU**
at 10 000 steps. The protocol's declared branch for this case is to report
the failure with its diagnostics rather than drop it silently, which is what
this section does.

**Consequence:** the capacity and geometry axes at 10 000 steps are untested,
and §3's recommendation supersedes them — the ladder at 30 000 steps is worth
more than Stage 2 was, and the width axis should be dropped or moved to a
larger card.

---

## 5. Scope and caveats

- **FID's floor here is 71.31**, not 0, and 512-sample FID is biased upward.
  Only arms measured identically are comparable; these numbers are not
  comparable with published FIDs.
- Two seeds, declared in advance. The 30 000-step seeds agree (235.86,
  228.42); the 10 000-step seeds do not (248.98, 269.02), so the curve's
  middle is less certain than its endpoints.
- The monotone fall is in medians. Seed 1 is monotone throughout; seed 0 is
  monotone except that 3 000 ≈ 600.
- Every arm carries R11, so this measures the corrected recipe, not the
  paper's.
- Run entirely on GPU (arithmetic validated to 9.4e-7, `device_check.json`).
- CIFAR-32 at 512-sample FID bounds the question; it does not settle it
  against the paper's ImageNet ladder.

## 6. Reproduce

```powershell
uv run --python 3.12 `
  --extra-index-url https://download.pytorch.org/whl/cu126 `
  --index-strategy unsafe-best-match `
  --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 `
  --with numpy --with scipy `
  python -m numerics.encoder_independent_drifting.run_phase16 `
  --seeds 2 --device cuda
```
