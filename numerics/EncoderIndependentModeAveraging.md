# Encoder-Independent Kernel Drifting — the mode-averaging research pass

## What the samples look like, why, and the one lever that moves it

*Research pass before the Phase 22 overnight run. Code: `diagnose_phase20.py`
(metric + render), `diagnose_phase21.py` (ESS ladder). Artifacts:
`phase20_probe.json`, `phase21_probe.json`, sample PNGs. Four permanent unit
tests added (123 total).*

---

## 1. The thing eighteen phases never did

**Nobody rendered a sample.** Phase 20 did, at the best configuration this
program has produced (Phase 19 arm D, 15 000 steps, FID 206.47):

*`phase20_samples_generated.png` — smooth low-frequency colour swirls. No
objects. No edges beyond blur boundaries. `phase21_samples_ess0p90.png` shows
the same at the operating point Phases 7–18 used: high-contrast abstract
blobs.*

**Every FID in Phases 16–19 was ranking blobs.** That does not retract those
measurements — the comparisons were sound — but it reframes what they were
comparing.

---

## 2. Two hypotheses I built probes for, and killed

**The metric is *not* the bottleneck.** I expected finite-sample FID noise to
explain Phase 19's unresolvable screen. Measured on one fixed generator with
everything else held constant:

| n | FID floor | FID (generator) | KID (generator) |
|---:|---:|---:|---:|
| 512 | 70.65 ± 0.80 | 206.47 ± 4.12 | 0.14200 ± 0.00450 |
| 1024 | 43.02 ± 0.68 | 189.87 ± 2.78 | 0.14282 ± 0.00305 |
| 2048 | 23.08 ± 0.30 | 177.41 ± 1.24 | 0.14210 ± 0.00113 |

Measurement sd at n = 512 is **4.12 FID** against Phase 19's within-arm seed
sd of **26.68** — about 2% of the variance. **Phase 19's noise is real
training variance.** Seeds genuinely converge to generators 60 FID apart.

The table did change the *metric*: FID's floor is 70.65 → 23.08 as n grows,
so almost all of the "floor" was finite-sample bias and the generator's own
FID is inflated ~29 points at n = 512. **KID is flat to three decimals**
across a 4× range, as an unbiased U-statistic should be. KID is primary from
here; FID is kept at both 512 and 2048 for continuity.

**"All targets collapse to the global mean" is refuted.** I added a
`teacher_spread` metric to test it; it reads ~1.0 at every rung. Targets stay
as spread out as the real positives. Each target is *individually* blurry
while the set stays diverse — the metric measured diversity, not blur, and
the finding belongs to realized ESS instead.

---

## 3. The mechanism, and the lever

`kernel_gradient._paper_side` builds the drift as a bi-softmax weighted
**average of the positives**. So every training target is an average of real
images, and the row effective sample size sets how many. An average of dozens
of CIFAR images is a blob — which is exactly what renders.

The ESS ladder (3 000 steps, 1 seed, cosine LR, R11):

| calibrated ESS | realized | images averaged | FID | tail |
|---:|---:|---:|---:|---:|
| 0.90 | 0.950 | 60.8 | 221.49 | 0.1021 |
| 0.50 | 0.849 | 54.3 | 216.68 | 0.1107 |
| 0.20 | 0.757 | 48.4 | 211.37 | 0.0973 |
| **0.05** | 0.609 | 39.0 | **204.16** | 0.0439 |
| 0.02 | 0.455 | 29.1 | 236.38 | 0.0088 |

**Sharpening from the 0.9 that Phases 7–18 used to 0.05 is worth 17 FID**,
monotone across four rungs with a clean reversal at 0.02 and no row collapse
anywhere.

The whole history now reads as one axis:

- Phase 7 pushed ESS **up** to 0.9 (≈58 images averaged) because it scored
  4.9× better on ED²;
- the metric audit then showed ED² is saturated by matching two moments — and
  **averaging preserves moments exactly while destroying structure**, so ED²
  was guaranteed to reward the blur;
- Phase 19 found 0.5 beats 0.9 on FID, the same axis pointing back;
- nobody had ever looked at the sharp side.

---

## 4. Why sharpening cannot finish the job

**Calibrating to 0.05 realizes 0.609 — a 12× gap.** The bandwidth is
calibrated target-only by design, and that calibration does not transfer to
cloud-vs-positive geometry: a point off the data manifold is nearly
equidistant from every real image, so the weights stay near-uniform whatever
the bandwidth. The trace shows realized ESS stable through training, so this
is structural, not transient.

That is **distance concentration** — the curse-of-dimensionality failure of
raw pixel kernels that motivates using an encoder in the first place.

And the sharp end is bounded by **arithmetic**, not by the mechanism.
Measured at 600 steps:

| calibrated ESS | realized | images | collapsed rows | affinity median |
|---:|---:|---:|---:|---:|
| 0.050 | 0.583 | 37.3 | 0.0000 | 8.5e-03 |
| 0.020 | 0.429 | 27.5 | 0.0000 | 3.4e-03 |
| 0.010 | 0.287 | 18.4 | 0.0000 | 1.1e-03 |
| **0.005** | **nan** | **nan** | **1.0000** | **0.0** |

At 0.005 `exp(-d/tau)` underflows in float32, every row dies, and the
denominator floor — not the data — decides each update. **0.010 is the
sharpest verified-viable setting.**

An earlier draft of the Phase 22 protocol specified 0.005 and would have run
two degenerate arms all night. The runner now prints
`*** KERNEL COLLAPSED ***` on any non-finite or collapsed row.

---

## 5. What was built

- **KID** (`fid.kid_from_features`), unbiased at every sample count, plus
  `frechet_from_features` so a probe can score many subsamples without
  re-running Inception.
- **Bandwidth mixtures** (`fixed_features.bandwidth_mixture`,
  `kernels.geometric_multipliers`, `tau_multipliers=`): the same block
  presented at several scales, positive-definite through the existing sum
  rule, ESS solved over the mixture as a whole. Verified: ESS hits target
  exactly at 3 and 5 levels, min eigenvalue positive, tau ratios exact.
  4 permanent tests.
- **Cost benchmark** (30 000-step run): base 0.31 h, mixture 0.35 h,
  positives 256 0.35 h, **cloud 1024 1.55 h**. The mixture and positive count
  are nearly free; cloud is 5× and would spend a night on one axis.

---

## 6. The correction that shaped the overnight design

My first arm set raised positives to 256 at fixed ESS *fraction*, with the
stated rationale that a larger attractor set "makes genuine sharpness
estimable". The fraction is roughly scale-free, so the *count* went from 42
images to **168** — the arm did the opposite of its purpose.

The real question is whether sharpening past the point where Phase 21
reversed becomes viable once there are more positives to select among: at 64
positives, averaging 29 of them is 45% of the batch and barely selective; at
256 the same count is 11%. Phase 22's `C_sharper` vs `E_sharper_pos` is that
test, with `D_pos` separating "more positives" from "sharper".

---

## 7. The identified next step

**Log-space stabilization of the bi-softmax.** Computing it through
`logsumexp` with a row-max shift removes the underflow floor entirely and
opens the regime where each target averages a *handful* of images — which is
where the mechanism hypothesis actually lives, and which no bandwidth choice
can currently reach.

Not attempted in this pass: it changes the core field, it needs an
equivalence test against the current implementation in the healthy regime,
and it is not something to write and launch unattended.

---

## 8. Scope

- One render, one seed, 3 000 steps for the ESS ladder. The ladder's monotone
  trend over four rungs is stronger than any single contrast in Phase 19, but
  it is still one seed; Phase 22 arm B vs A is what tests it.
- The underflow bound is float32-specific and measured at one resolution and
  one kernel (`smooth_laplace`).
- Raw geometry only; licenses no claim about encoder dependence.
- Still not the paper's method: pixel-space drift with feature-space kernel
  weights.
