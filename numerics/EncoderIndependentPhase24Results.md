# Encoder-Independent Kernel Drifting — Phase 24 preflight

## The latent space has a good ceiling and the drift cannot reach it. Both of my arguments were wrong, and the preflight cost 40 minutes instead of 8 hours.

*Code: `appearance.py`, `autoencoder.py`, `latent_drifting.py`,
`models.LatentGenerator`, `diagnose_phase23.py`, `diagnose_phase24.py`.
Artifacts: `phase23_probe.json`, `phase24_probe.json` (+ `.sha256`), grids.*

---

## 1. What passed: the instruments

`diagnose_phase23.py` validates the new measures against known degradations —
**11/11 declared checks pass**. It also cost me three wrong predictions, all
the same mistake, and the consolidated finding is worth more than the checks:

**Precision measures typicality, not realism.**

| input | precision | recall | caught by |
|---|---:|---:|---|
| real | 0.726 | 0.748 | — |
| moment-matched Gaussian | **0.842** | 0.000 | recall |
| box blur 7 | **0.681** | 0.000 | recall |
| noise 0.3 | 0.350 | 0.009 | both |
| shuffled pixels | 0.029 | 0.000 | both |

A structureless Gaussian scores precision *above real data*. Precision only
falls for things that leave the typical set. **Precision detects artifacts,
recall detects blur and collapse, neither is readable alone.**

Also found: the avg-pool-then-*nearest*-upsample "blur" used elsewhere in this
package is a downsample, not a low-pass — its block edges *add* high-frequency
energy. `diagnose_phase18.py` uses it, so that probe's blur column was partly
measuring blockiness.

---

## 2. What passed: the ceiling

| latent d | KID | FID2048 | precision | recall | alpha | ae mse |
|---|---:|---:|---:|---:|---:|---:|
| 128 | +0.11600 | 118.30 | 0.638 | 0.069 | 4.910 | 0.01938 |
| 256 | +0.06808 | 78.69 | 0.716 | 0.268 | 4.856 | 0.01116 |
| **512** | **+0.03064** | **49.44** | **0.736** | 0.496 | 4.699 | 0.00650 |
| *real* | −0.00004 | 22.50 | 0.731 | 0.751 | 3.603 | — |

*(pixel-space best, Phase 22 `F_mix`: KID +0.13116)*

At d=512 the ceiling is **4× better on KID** than anything pixel space
achieved, precision **exceeds real data's**, and the reconstruction grid shows
**recognizable objects** — cars, horses, birds, planes, deer. The decoder is
not the problem.

---

## 3. What failed: argument one — selectivity

I argued in `EncoderIndependentPhase23Research.md` §2 that both Phase-22 walls
came from estimating a kernel in 3072 dimensions from 64–256 samples, and
predicted a code space would sharpen selectivity.

Measured in the **actual field configuration** (generator cloud vs real
positives, target ESS 0.05):

| space | dim | realized ESS | images averaged | cloud-vs-cloud CV |
|---|---:|---:|---:|---:|
| pixel | 3072 | 0.681 | 43.6 | 0.0812 |
| code | 256 | 0.920 | 58.9 | 0.0975 |
| code | 512 | 0.944 | 60.4 | 0.0834 |

**The code space is less selective, not more.** And on real-vs-real data the
same calibration hits its target almost exactly (0.048 vs 0.050) *in both
spaces* — so ambient dimension was never the obstacle.

The last column is the actual cause: **the generator's cloud has CV ≈ 0.08 in
every space** against real data's 0.22–0.30. Its outputs are nearly identical
to one another, so from any cloud point every real positive sits at nearly the
same distance and the weights flatten. Cutting 3072 → 256 makes this worse,
because a more compact code space lets the common offset dominate more.

---

## 4. What failed: argument two — averages decoding onto the manifold

The load-bearing claim did not depend on selectivity: *the average of many
codes decodes back onto the image manifold, while the average of many images
does not.* A 44-sample average is what the pixel field actually forms.

| teacher | KID | precision | recall | alpha |
|---|---:|---:|---:|---:|
| pixel average of 44 | +0.33368 | 0.795 | 0.000 | 3.681 |
| **decoded code average of 44** | **+0.42263** | **0.352** | 0.000 | 4.580 |
| real reference | — | 0.726 | 0.748 | 3.603 |

**The decoded code average is worse on every measure**, precision 0.352 against
0.795. `phase24_teacher_code_average.png` is uniform brown-grey mush.

The reason is straightforward in hindsight: a plain autoencoder's latent space
is trained for *compression*, not convexity. Nothing makes the midpoint of two
codes decode to anything sensible. That property belongs to a KL-regularized
or otherwise distribution-matched latent space, and I assumed it without
checking.

---

## 5. The mechanism, properly identified

Both failures point at the same thing, and it is not dimension and not the
space:

> The teacher is a **near-uniform average of ~44 samples** because the
> generator's cloud is concentrated (CV 0.08). A near-uniform average of 44
> images is **hyper-typical**: precision 0.795 with recall 0.000 — the
> moment-matched-Gaussian signature. The generator regresses onto hyper-typical
> blurs, so its cloud stays concentrated, so the teacher stays a blur.

It is a self-reinforcing loop, and it explains the program's older
"self-reference" finding (`T = f + ηV` anchors the generator to its own cloud)
which had only ever been measured in ED² and the tail.

**Global bandwidth calibration cannot break it.** The calibration is
target-only by design, so it achieves its declared ESS on real-vs-real data
(0.048 vs 0.050) and then realizes 0.68–0.94 against the actual cloud. The
knob every phase since Phase 7 has been tuning does not control the quantity
it is believed to control.

---

## 6. What to do — the fix the measurement implies

**Per-row adaptive bandwidth.** Instead of one global τ calibrated on target
data, solve for τ_i *per cloud point* so that row i realizes the declared ESS.
This is the perplexity calibration of t-SNE/UMAP, it is cheap (a bisection per
row), and it directly repairs the defect measured in §3 and §5: realized
selectivity would equal the target by construction instead of drifting to
0.68–0.94.

It attacks the loop at the point where it is actually broken, and it is
testable **in pixel space** — the space question is now decoupled and should
not be bundled in.

Deprioritized on this evidence:

- **Latent drifting as argued here.** Both rationales are refuted. It should
  not be run at length until some mechanism gives the code space the convexity
  §4 shows it lacks.
- **A VAE or otherwise regularized latent space** is the honest repair of §4
  and is a genuine candidate, but it is a second structural change and should
  wait until §6's fix is measured.

---

## 7. Scope and cost

- CIFAR-10 32×32; comparable only between arms measured identically.
- §3 and §4 are single-configuration probes on untrained generators and real
  batches respectively, not seeded comparisons. They are large, mechanistic
  effects (precision 0.795 → 0.352), not the ~15 FID differences that need
  eight seeds — but they are screens, and a surviving version of either
  argument would need a seeded test.
- The autoencoders are converged by reconstruction MSE (§2 history), trained on
  the train split only, and never see eval data.
- **This preflight cost ~40 minutes and cancelled an 8-hour run whose
  rationale it refuted.** Phase 22 spent 7.31 h establishing a ceiling that
  arithmetic could have predicted; this is the correction to that habit and
  the practice worth keeping.
