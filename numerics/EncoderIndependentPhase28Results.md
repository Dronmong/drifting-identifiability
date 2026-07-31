# Encoder-Independent Kernel Drifting — Phase 28 results

## The generator can produce recognizable images. The drifting objective destroys them in 1500 steps.

*Code: `diagnose_phase28.py`. Artifacts: `phase28_probe.json`,
`phase28_warm512.json` (+ `.sha256`), grids. 5 000-step reference, 512
particles, 12 000 memorization steps, 6 000 drifting steps.*

---

## 1. Results

| stage | KID | precision | recall | alpha | looks like |
|---|---:|---:|---:|---:|---|
| real | ~0 | 0.767 | 0.767 | 3.61 | — |
| **A memorize n=512** | **+0.06110** | 0.791 | **0.224** | 4.417 | **recognizable objects** |
| A memorize n=2048 | +0.19587 | 0.850 | 0.000 | 4.431 | past capacity |
| 0 AE recon, start | +0.07962 | 0.633 | 0.301 | 4.858 | recognizable |
| 0 AE recon, +40 iters | +0.08599 | 0.625 | 0.270 | 4.845 | slowly eroded |
| B cold, +6000 drift | +0.14959 | 0.523 | 0.000 | 4.503 | worms |
| **B warm(n=512), +6000 drift** | **+0.16538** | 0.648 | **0.000** | 4.445 | **worms** |

**The corrected Stage B is the result:**

```
step    0   KID=+0.06138  P=0.768  R=0.230  alpha=4.415   recognizable objects
step 1500   KID=+0.17233  P=0.627  R=0.000  alpha=4.391
step 6000   KID=+0.16538  P=0.648  R=0.000  alpha=4.445
```

Drifting takes a generator holding **KID 0.061 and recall 0.230** with visibly
recognizable CIFAR objects and, within **1500 steps**, drives it to **KID 0.172
and recall 0.000** — then holds it there.

---

## 2. The finding

> At the **same architecture** and the **same spectral smoothness**
> (alpha ≈ 4.42), fixed-pairing regression reaches recall 0.224 / KID 0.061 with
> recognizable objects, while the drifting objective reaches recall 0.000 /
> KID 0.12–0.17. **The entire gap is coverage, and it is the objective's doing.**
> The generator was never the bottleneck.

Note what does *not* change during the destruction: **alpha holds at 4.39–4.45
throughout** while recall goes 0.230 → 0.000. The objective does not blur the
output. It removes coverage.

### The mechanism this points to: parameter sharing

Phase 26 showed the data distribution is stationary for **free particles** — real
data survived 40 iterations with KID at the floor. Phase 28 shows a **generator**
near a good state is destroyed by the same field. The two are not in conflict;
the difference is the only thing that differs:

Free particles move independently, so each can sit at its own local fixed point.
A generator must satisfy every teacher **with one shared set of parameters**.
The teachers are kernel-weighted averages of ~40 real images (Phase 22/25), and
the only function that fits a family of such averages across the whole latent
space is one that collapses coverage. **The field is pointwise reasonable and
parametrically unlearnable.**

That is consistent with every prior measurement — the zero-recall teacher of
Phase 25, the hyper-typicality everywhere, the bandwidth invariance — and it is
the first account that explains why free-particle and generator results diverge.

---

## 3. Corrections to earlier phases

**Phase 27's alpha framing is wrong.** I said the unrecoverable direction was
loss of high-frequency content, measured by alpha. But recognizable images sit
at alpha 4.4–4.9 (memorization 4.417; AE recon 4.858), essentially the same as
the drifting output's 4.43–4.49. **alpha does not measure image quality.** The
Phase 27 measurements stand; the interpretation should be in terms of coverage.

**The auto-verdict in this probe is wrong and must not be quoted.** It printed
*"the generator cannot emit sharp images … the field is exonerated, and 27 phases
tuned the wrong component"* — the exact opposite of the truth. Cause: I declared
`alpha < 4.0 and recall > 0.30` as "sharp", picking 4.0 from intuition without
calibrating against any measured reference. Real data is 3.61 and *every*
recognizable state in this program sits above 4.4, so the threshold could never
have passed.

**This is the fourth declared threshold I have set from intuition rather than
from a measured anchor** (after Phase 19's 4/4 sign rule, Phase 23's three
precision/blur expectations, and Phase 27's `basin_is_wide` ignoring the blur
veto). The pattern, not the individual slips, is the problem: **a threshold must
be calibrated against a reference state measured in the same units before it is
declared.** Phase 23's ladder exists precisely for this and I did not use it
here.

**A design flaw in Stage B.** It warm-started from `max(MEMORY_SIZES)` = 2048 on
the assumption that more images make a better start. The measurement shows 2048
is past capacity (recall 0.000), so the original Stage B compared two bad
starts and reported a misleadingly mild effect (warm 0.122 vs cold 0.150). The
corrected run from n=512 is what §1 reports.

---

## 4. Where this leaves the program

**The drifting objective, as specified, cannot train a parametric generator to
cover this data.** That is now measured three ways: the teacher has zero recall
at every bandwidth (Phase 25), the map erodes every high-alpha state it is given
(Phases 27, 28 Stage 0), and it destroys a demonstrably good generator state in
1500 steps (Phase 28 Stage B).

**And the encoder question is now answerable in a way it was not before.** The
encoder-independence result (Phases 17–18: raw pixels beat a pretrained encoder
by +138 FID) was always caveated as "a statement about a method that does not
work". Phase 28 identifies *why* it does not work, and the cause — parametric
collapse under kernel-averaged teachers — is independent of which geometry
supplies the kernel. That strengthens the negative result: no encoder choice
could have rescued it.

**What would be worth testing, if anything:**

1. **A non-averaged teacher.** ~~The mechanism in §2 blames the convex-combination
   form. An assignment-based teacher (each cloud point matched to *one* real
   sample, Hungarian or Sinkhorn-hard) is not an average and would test the
   mechanism directly. This program has Sinkhorn machinery already.~~

   **Superseded by `EncoderIndependentPhase29Research.md`. Phase 13 already ran
   this**, and it refines §2's mechanism rather than confirming it. An exact
   Hungarian bijection with **fresh** latents gives the worst spectral tail of
   any arm ever measured here (0.0006) despite perfect non-collision, while the
   same assignment with **frozen** latents works. Nearest-neighbour collapses to
   `distinct = 1` target with second moment exactly 0.000. So averaging was
   never the operative property: **the operative property is whether the
   latent→target correspondence is stable.** With fresh latents the target is a
   random variable and squared-error regression converges to its conditional
   mean — the population mean — however sharp each individual target was. That
   covers this phase's result and Phase 13's with one mechanism, and it explains
   why free particles (Phase 26) behave differently: no shared parameters, no
   latent indexing, no conditional mean to collapse onto.
2. **Report the memorization result as the harness's own ceiling.** KID 0.061
   with recall 0.224 from a fixed-pairing regression is what this generator can
   do. Every drifting number in 28 phases should be read against that, not
   against the Gaussian bar.

**Not worth testing:** any further bandwidth, selectivity, mixture, geometry or
encoder variation. §2 explains why none of them can matter.

---

## 5. Scope

- One seed per configuration; these are mechanism probes, not seeded
  performance claims. The Stage B effect is very large (recall 0.230 → 0.000,
  KID +0.104) and reproduced in direction by the n=2048 warm start.
- Stage A is a *memorization* test with a fixed latent-image pairing, evaluated
  on its own training pairs. It measures representational capacity, not
  generalization, and the recall 0.224 figure must not be read as a generative
  result.
- Without the fixed pairing the stage would be vacuous — the target would be
  independent of `z` and squared error minimized by the mean image. This is
  noted in the module docstring because it nearly went unnoticed.
- Still not the paper's method: pixel-space drift with feature-space kernel
  weights.
