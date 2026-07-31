# Encoder-Independent Kernel Drifting — Phase 22 protocol

## Can any variant of this mechanism produce image structure?

**Frozen before the run. Declared outcomes in §7. Nothing below is tuned
against a result.** Overnight run, ~7 h.

---

## 1. What this run is for

Phase 20 rendered the best generator this program has produced (arm D, 15 000
steps, FID 206) and it contains **no image structure** — smooth low-frequency
colour swirls, no objects, no edges. Every FID in Phases 16–19 was comparing
blobs.

Phase 21 found why, and found the first real lever:

| calibrated ESS | realized | images averaged | FID |
|---:|---:|---:|---:|
| 0.90 | 0.950 | 60.8 | 221.49 |
| 0.50 | 0.849 | 54.3 | 216.68 |
| 0.20 | 0.757 | 48.4 | 211.37 |
| **0.05** | 0.609 | 39.0 | **204.16** |
| 0.02 | 0.455 | 29.1 | 236.38 |

The drift is a bi-softmax weighted **average of the positives**, so every
target is an average of real images and the row ESS sets how many. Sharpening
from the 0.9 that Phases 7–18 used to 0.05 is worth **17 FID** at 3 000
steps, monotone across four rungs, with a clean reversal at 0.02 and no row
collapse anywhere.

**But calibrating to 0.05 realizes 0.609** — a 12× gap. Target-only
calibration does not transfer to cloud-vs-positive geometry, because a point
off the data manifold is nearly equidistant from every real image. Even the
sharpest working setting still averages **29 images**. That is distance
concentration, and no bandwidth escapes it.

So this run asks whether the axes that attack concentration *directly* can do
what bandwidth alone cannot.

---

## 2. Arms

All at 30 000 steps, raw geometry, cosine LR, R11 scalar teacher, no EMA
(refuted in Phase 19), cloud 256, width 64.

| arm | target ESS | bandwidth levels | positives | why |
|---|---:|---:|---:|---|
| `A_control` | 0.500 | 1 | 64 | the Phase-19 winner — control |
| `B_sharp` | 0.050 | 1 | 64 | Phase 21's optimum, now with seeds |
| `C_sharper` | 0.010 | 1 | 64 | past where Phase 21 reversed |
| `D_pos` | 0.050 | 1 | 256 | more attractors, same bandwidth |
| `E_sharper_pos` | 0.010 | 1 | 256 | **the interaction this run exists for** |

Realized images averaged, measured at 600 steps before freezing this table:
54 / 37 / **18** / 140 / 59 / 51. The arms span an **18× range on the axis
the mechanism says matters.**

**The sharp end is bounded by arithmetic, not by the mechanism.** ESS 0.005
was measured **dead** — collapsed row fraction 1.0, affinity median exactly
0, half the denominators pinned to the floor — because `exp(-d/tau)`
underflows in float32 before the kernel becomes genuinely selective. 0.010 is
the sharpest verified-viable setting. An earlier draft of this protocol
specified 0.005 and would have run two degenerate arms all night; the runner
now prints `*** KERNEL COLLAPSED ***` on any row with a non-finite or
collapsed ESS so this can never be reported quietly.

**This makes log-space stabilization the highest-value follow-up.** Computing
the bi-softmax through `logsumexp` with a row-max shift would remove the
underflow floor entirely and open the regime where each target averages a
*handful* of images — which is where the mechanism hypothesis actually lives.
It is deliberately **not** attempted here: it changes the core field, it needs
an equivalence test against the current implementation in the healthy regime,
and it is not something to write and launch unattended.
| `F_mix` | 0.050 | 5 | 64 | multi-scale kernel |

**The parametrization matters and I got it wrong first.** The ESS *fraction*
is roughly scale-free, so holding it while raising the positive count raises
the *count* of images averaged — measured, 42 at 64 positives and **168 at
256**. More positives at a fixed bandwidth is therefore **blurrier**, not
sharper. An earlier version of this arm set claimed 256 positives would make
"genuine sharpness estimable" while doing the opposite.

The real question is whether sharpening *past the point where Phase 21
reversed* (0.02 → 29 of 64 images, FID 236) becomes viable once there are
more positives to select among. At 64 positives, averaging 29 of them is
still 45% of the batch — barely selective. At 256 positives the same count is
11%. **`C_sharper` versus `E_sharper_pos` is that test**, and `D_pos` is the
control that separates "more positives" from "sharper".

**Why these and not cloud size.** Measured cost per 30 000-step run: base
0.31 h, mixture 0.35 h, positives 256 0.35 h, **cloud 1024 1.55 h**. The
mixture and the positive count are nearly free and both are untested in FID;
cloud is 5× and would consume the night on one axis. Cloud remains declared
untested, not dismissed.

**Why a bandwidth mixture.** A single bandwidth sees one length scale, which
is the standard motivation for mixtures in the MMD generative literature
(GMMN; MMD-GAN; KID). Here it has a sharper rationale: distance concentration
is scale-dependent, so presenting several scales at once is the cheapest
attack on the mechanism Phase 21 identified. Multipliers are the declared
geometric ladder `(0.25, 0.5, 1, 2, 4)`, never searched, and the ESS solve
runs over the mixture as a whole.

**Prediction on the mixture, declared now.** The measured realized ESS rises
when the mixture is on (0.657 → 0.803 at 64 positives), because a symmetric
span adds *wider* kernels that flatten the weights. If sharpness is what
helps, the mixture should **hurt**. It is run as the literature defines it
rather than pre-tilted downward, so that a negative result means something.

**4 seeds**, paired within seed: every arm shares a seed's generator
initialization, latent stream and target order.

---

## 3. Metrics

**KID is primary.** Phase 20 measured FID's floor at 70.65 / 43.02 / 23.08
for n = 512 / 1024 / 2048 — almost all of the "floor" was finite-sample bias,
and the generator's own FID is inflated ~29 points at n = 512. KID over the
same draws read 0.14200 / 0.14282 / 0.14210: **flat to three decimals**, as
an unbiased U-statistic should be.

FID is reported alongside at n = 512 for continuity with Phases 16–19, and
**at n = 2048** so at least one number in this program is not bias-dominated.

**Sample grids are a first-class output, not a garnish.** Phase 20 showed FID
206 and FID 221 are both blobs; a number that cannot see the difference
between blobs and images cannot be the only readout. One grid per arm.

---

## 4. Statistics

Phase 19's 4-of-4 sign test was a bad rule: at 4 seeds it is a 12.5%-level
test that a single reversal destroys. **This run reports paired mean, sem and
a paired t per contrast, and calls nothing established below p < 0.05.** The
standing project rule is ≥ 8 seeds for ~15 FID effects; 4 seeds here is a
deliberate trade for covering five arms, and §7 states what that costs.

Contrasts declared in advance:

- `sharp` = B − A (does Phase 21 survive seeds and a 10× budget?)
- `mixture` = C − B and E − D
- `positives` = D − B and E − C
- `best − A` (total available gain)

---

## 5. What would count as image structure

FID and KID cannot answer the question this run is named for, so the
structural readout is declared here rather than eyeballed afterwards:

- **spectral tail** toward real data's ~0.13 (Phase 21's sharpening drove it
  the *wrong* way, 0.102 → 0.0088);
- a **visible edge** in the rendered grid — anything with a boundary that
  is not a blur gradient.

Neither is a gate. Both are reported.

---

## 6. Cost

5 arms × 4 seeds × 30 000 steps at 0.31–0.35 h ≈ **6.8 h**, plus Inception
scoring. Fresh seed block `SEED_OFFSET = 35000`, disjoint from every prior
phase.

---

## 7. Declared outcomes

- **Sharpening replicates and an axis adds to it** → the mechanism has real
  headroom; take the winner to a long-budget scaling run and report the ESS
  finding as this program's first genuine improvement.
- **Sharpening replicates, nothing adds** → ESS was the whole lever;
  distance concentration bounds the method, and the honest output is the
  mechanism result plus the bound.
- **Sharpening does not replicate at 30 000 steps** → Phase 21 was a
  short-budget artifact; the ESS ladder is retired and this program's
  configuration work is finished.
- **Structure appears in any grid** → the biggest result the program could
  produce, and it changes what is worth running next regardless of FID.
- **No grid shows structure at any setting** → *the strongest likely
  outcome*: raw-pixel kernel drifting cannot generate images at this scale,
  the encoder-independence result stands but is about a method that does not
  work, and the program should be written up as a negative with a mechanism.

---

## 8. Scope

- CIFAR-10 32×32. Only arms measured identically are comparable; no number
  here is comparable with a published FID.
- 4 seeds — below this program's own ≥ 8 rule (§4). Effects near 15 FID may
  come back unresolved, and that is a known cost of covering five arms.
- The ESS ladder that motivates this run was **one seed at 3 000 steps**.
  Arm B versus A is what tests it properly.
- Raw geometry only; licenses no claim about encoder dependence.
- Still not the paper's method: pixel-space drift with feature-space kernel
  weights.
