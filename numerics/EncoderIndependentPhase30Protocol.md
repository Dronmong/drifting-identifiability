# Encoder-Independent Kernel Drifting — Phase 30 protocol

## Does capacity or batch size unlock nonzero recall?

**Frozen before the run. Pre-flight recorded in `phase30_preflight.json`.**

---

## 1. The question

Twenty-nine phases produced **recall 0.000** under every fresh-latent objective
tested — drifting, nearest-neighbour, exact Hungarian bijection, IMLE in naive
and frozen forms, and Hungarian on frozen latents. The only configuration that
ever held coverage was memorization of a fixed 512-image set (recall 0.224),
which is not a generative model.

The generator is the one component never varied. Phase 28 showed it *can* hold
recall 0.224, and that its capacity is exhausted somewhere between memorizing
512 images (mse 0.011) and 2048 (mse 0.049, recall 0.000). So capacity is
plausibly binding for coverage, and it has never been tested.

**Success is declared as recall > 0.05**, calibrated against measured anchors,
not intuition: 0.000 observed across six independent objectives, 0.224 at the
memorization ceiling, 0.496 for an autoencoder reconstruction, 0.72–0.74 for
real data. This is the first threshold in this program set from a reference
measured in the same units — the previous four were set from intuition and all
four were wrong.

---

## 2. A logic correction that shapes the design

My stated hypothesis — "the collapse is variance-driven, and variance falls with
scale" — conflates two mechanisms:

- **batch size** reduces target variance, attacking the Phase 29
  correspondence-stability mechanism directly;
- **capacity** raises expressiveness and does *not* reduce target variance.

Both are real levers but they are different, so a single "scaled" arm cannot
attribute an effect. The design separates them.

---

## 3. What the pre-flight forced

The pre-flight returned **NO-GO** on the design I first proposed, for resource
reasons, and it repeated a failure already on record: Phase 16's `S2_wide`
(width 256, cloud 256) exhausted this card and died before writing its artifact.

Measured, one config per process:

| config | params | allocated | ms/step | h per 30k |
|---|---:|---:|---:|---:|
| w64 cloud256 | 147k | 595 MiB | 34.6 | 0.29 |
| w128 cloud256 | 515k | 2996 MiB | 111.5 | 0.93 |
| w192 cloud256 | 1.10M | 1723 MiB | 197.3 | 1.64 |
| **w256 cloud256** | 1.91M | **5972 MiB** | **3274** | **27.3** |
| w256 cloud128 | 1.91M | 3801 MiB | 210.0 | 1.75 |
| w384 cloud128 | 4.20M | 7296 MiB | 956 | 7.96 |

Width 256 at cloud 256 needs 5972 MiB on a 6141 MiB card, spills to system
memory over PCIe, and runs **95× slower** than width 64 — 27 h for one run, and
120 h for the factorial I originally described as an "overnight run."

**So width 256 with cloud 256 is out.** Cloud size is held at 256 to stay
comparable with the Phase 28/29 recall-0.000 reference, which caps capacity at
width 192 (7.5× the baseline parameters).

*Unresolved oddity, recorded rather than smoothed over:* w128 measured 2996 MiB
allocated against w192's 1723 MiB — non-monotone and unexplained. Both sit far
below the limit so it does not gate the run, but the memory figures should not
be treated as precise.

---

## 4. Design

| arm | width | params | positives | cloud | h per seed |
|---|---:|---:|---:|---:|---:|
| `w64_p64` | 64 | 147k | 64 | 256 | ~0.25 |
| `w64_p256` | 64 | 147k | 256 | 256 | 0.29 |
| `w128_p256` | 128 | 515k | 256 | 256 | 0.93 |
| `w192_p256` | 192 | 1.10M | 256 | 256 | 1.64 |

This gives a **capacity ladder** (64 → 128 → 192 at fixed positives) and a
**batch contrast** (`w64_p64` vs `w64_p256`) while keeping cloud size fixed.
`w64_p64` is the arm whose recall is already known to be 0.000, so it anchors
the comparison inside this run rather than across runs.

**Budget: 30 000 steps, 2 seeds. Projected ~6.2 h.**

Two seeds is below this program's standing ≥8-seed rule, and that is a declared
exception with a reason: the question is **categorical** (does recall leave
0.000, observed six times across independent objectives), not a ~15 FID
difference. Any *quantitative* recall comparison between arms in this run is
explicitly not licensed and must not be reported as one.

Held fixed: drifting objective, raw geometry, `target_ess_fraction = 0.05`
(legacy calibration — the operating point Phase 22 measured best, realizing
ESS ≈ 0.62–0.76), R11 scalar teacher, η = 0.5, **cosine learning rate** (matching
the Phase 28/29 reference; a constant schedule would break comparability),
Adam at 2e-3 peak, CIFAR-32, disjoint train/eval splits, no EMA.

Equal *steps* means unequal *compute* across arms. That is intended — the
question is whether capacity unlocks coverage, not whether it is efficient —
and the wall time per arm is reported so the asymmetry is visible.

---

## 5. Metrics

**Recall is primary.** KID, precision, alpha, spectral tail and second moment
are recorded. Precision is explicitly *not* a success signal: a moment-matched
Gaussian scores precision 0.867 with recall 0.000 in this very pre-flight, so
precision alone rewards typicality (Phase 23). Sample grids for every arm.

The pre-flight verified the instrument inside the same code path: real-vs-real
recall 0.737, moment-matched Gaussian recall 0.000. A recall of 0.000 in this
run therefore cannot be a broken measurement.

---

## 6. Declared outcomes

- **Any arm reaches recall > 0.05** → scale unlocks coverage. Report which lever
  did it, and the encoder ladder (already built) becomes a real experiment for
  the first time in the program.
- **Recall rises with capacity but no arm crosses 0.05** → the direction is
  right and the harness is under-scaled; the next step is a bigger card or a
  reduced cloud, and the claim becomes quantitative rather than categorical.
- **Recall stays 0.000 everywhere** → capacity and batch are not the
  obstruction. The remaining candidate is the correspondence-stability problem
  of Phase 29, which an encoder or amortized inference network solves and which
  would make encoder-independence a substantially harder claim than the geometry
  results suggest.
- **KID improves while recall stays 0.000** → the summary statistic tracks scale
  and coverage does not, which would be the fourth time in this program that KID
  and structure move independently.

---

## 7. Scope

- CIFAR-10 32×32; comparable only between arms measured identically.
- 2 seeds, justified in §4 for a categorical question only.
- Cloud size fixed at 256 and never varied in this program — `S2_cloud` died
  with `S2_wide` in Phase 16 and was never re-run.
- Capacity reaches 7.5× baseline, not the 100× that separates this harness from
  a realistic generator. A null result bounds *this* range, not scale in general.
- Still not the paper's method: pixel-space drift with feature-space kernel
  weights.
