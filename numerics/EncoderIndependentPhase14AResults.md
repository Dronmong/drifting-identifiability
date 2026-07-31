# Encoder-Independent Kernel Drifting — Phase 14A results

## The precondition fails, and the reason invalidates how the program has been measuring

*Plan: `EncoderIndependentPhase14Plan.md`. Code: `run_phase14a.py`,
`encoders.py`, `fid.py`. Artifacts: `phase14a_uncorrected.json`,
`phase14a_r11.json` (+ `.sha256`). 3 seeds, 600 steps, CIFAR-10 at 32×32,
run entirely on GPU. First use in this program of a **real pretrained
encoder** and of **FID**.*

---

## 0. Summary

14A asked whether this harness reproduces the paper's finding that quality
tracks encoder quality. **It does not — and the calibration explains why in a
way that matters far beyond this phase.**

| geometry | FID | ED² | 2nd moment |
|---|---:|---:|---:|
| G5 degraded | **255.0** | 0.513 | 0.974 |
| G4 raw pixels *(encoder-free)* | 264.4 | **0.373** | 0.998 |
| G3 autoencoder | 278.4 | 0.376 | 1.015 |
| G2 random ResNet18 | 280.2 | 0.504 | 1.011 |
| **G1 pretrained ResNet18** | **302.1** | **0.754** | 0.936 |

*(all arms carrying R11, so second moments are healthy and the comparison is
not confounded by the deficit — see §2)*

The ranking is **exactly inverted** from the paper's ablation: a real
pretrained encoder is the *worst* geometry, a deliberately degraded one is
the best by FID, and raw pixels are best by ED².

---

## 1. The calibration that settles the interpretation

A 255–302 spread means nothing without knowing the scale. Measured at the
same 512 samples:

| reference | FID |
|---|---:|
| real vs real — **the floor** | **70.7** |
| train split vs eval split | 72.4 |
| the five generator arms | 255 – 302 |
| **free particles, 600 steps** | **381.2** |
| pure noise — **the ceiling** | **416.9** |

So the usable range is 70.7 to 416.9, and every arm sits **53–67% of the way
from real data to pure noise.** Nothing here is in the regime the paper's
ablation lives in — its ladder runs down to FID 3.36, where images are
recognizable and geometry can plausibly matter. **These arms are comparing
which geometry is least bad at producing near-noise**, and that comparison
carries no information about the thesis.

The precondition fails: **the harness cannot currently test whether the
encoder matters**, because no arm is close to the quality regime where it
could.

---

## 2. The confound I introduced, and removing it

The first ladder ran without R11 and produced the same inverted ranking — but
for a different and misleading reason:

| geometry | FID (no R11) | ED² | **2nd moment** |
|---|---:|---:|---:|
| G4 raw pixels | 258.5 | 2.89 | **0.296** |
| G2 random ResNet | 292.5 | 5.52 | **0.133** |
| G1 pretrained ResNet | 298.4 | 10.55 | **0.022** |

The quality ranking was simply the *collapse* ranking. That is the Phase-1/2
mistake — comparing geometries on a baseline broken for reasons unrelated to
geometry — and I repeated it. Both ladders are kept: the corrected one (§0)
is the comparison, the uncorrected one is preserved as
`phase14a_uncorrected.json`.

**It did surface something new.** The second-moment deficit is far worse
under encoder geometry — **0.022 for pretrained ResNet against 0.296 for raw
pixels**, a 13× difference — and R11 rescues it by 14× (ED² 10.55 → 0.75).
The program had only ever measured the deficit on raw pixels; it is
geometry-dependent, and worst exactly where the features are most semantic.

---

## 3. The finding that reaches back through the whole program

**ED² and FID disagree in rank order, and most sharply on the comparison the
last seven phases were built on.**

| | ED² | FID |
|---|---:|---:|
| free particles | **~0.07–0.2** *(best in the program)* | **381.2** *(near-noise)* |
| the generator arms | 0.37 – 0.75 | 255 – 302 |

Free particles are the best thing this program has produced by energy
distance and among the *worst* by FID. Phases 6C, 7B, 10, 12 and 13 all used
the particle cloud as the quality reference the generator should approach.
**Under a semantic metric that reference is near-noise.**

Energy distance and the normalized composite are **pixel-space** statistics.
A cloud can match the pixel-space distribution well while carrying no
recognizable image content, and that is what the particle system appears to
be doing. Nothing in thirteen phases would have detected this, because
nothing in thirteen phases used a semantic metric.

This does not retract the mechanism results — the shape law, the
self-reference finding, R11's replications are all statements about the
second moment and ED², and they stand as such. What it removes is the
inference from "better ED²" to "better generative model".

---

## 4. What this changes

**Immediate:** 14B as planned is not worth running. Comparing encoder-free
against encoder-based at FID ~300 would produce a number with no
interpretation, which is exactly what 14A existed to prevent.

**The blocking problem is quality, not geometry.** To test the thesis at all,
the arms must reach a regime where images are recognizable — FID moving
toward ~100 rather than sitting at ~280. That is a scaling problem: 600 steps
with a width-64 generator is roughly three orders of magnitude short of the
paper's training. The GPU makes this approachable for the first time.

**Recommended next step — a scaling run, not a mechanism run.** Take the best
encoder-free configuration (raw pixels, ESS-0.9, R11) and train it far
harder — 10⁴–10⁵ steps, larger generator, larger field cloud — with FID
tracked as the primary readout, and find where FID plateaus. Two outcomes,
both decisive:

- **FID falls to a regime where images are recognizable** → the ladder
  becomes meaningful, 14B runs, and the thesis is finally testable;
- **FID plateaus near 300 regardless of budget** → drifting on raw pixels
  does not produce semantic content at this scale, which is a real and
  reportable answer about encoder-free drifting, and it would explain the
  paper's encoder dependence directly.

**Second, and cheap:** re-score a few historical arms with FID. If ED²
improvements from earlier phases (R11's 3.1×, the bandwidth's 4.9×) do not
move FID, then the program's headline reforms improve a pixel statistic and
not the generative model — which needs to be known before anything is
written up.

---

## 5. Scope and caveats

- **FID at 512 samples has a floor of 70.7**, not 0, and is biased upward.
  Comparisons between arms measured identically are sound; the absolute
  values are not comparable with published FIDs.
- ResNet18/ImageNet is a supervised classifier, not the paper's
  self-supervised encoder. It is a real pretrained semantic encoder and a
  large step up from an autoencoder stand-in, but the substitution stands
  behind every G1 number.
- Encoder features are L2-normalized and the encoder sees images upsampled to
  128; both are declared in `encoders.py` and neither was swept.
- All arms use the same target-ESS-0.9 calibration per geometry, and all were
  R15-admissible (ESS 0.947–0.992, no collapsed rows).
- Run entirely on GPU; arithmetic was validated against CPU to 9.4e-7
  (`device_check.json`). These numbers should not be spliced into CPU-era
  tables without saying so.

## 6. Reproduce

```powershell
uv run --python 3.12 `
  --extra-index-url https://download.pytorch.org/whl/cu126 `
  --index-strategy unsafe-best-match `
  --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 `
  --with numpy --with scipy `
  python -m numerics.encoder_independent_drifting.run_phase14a `
  --seeds 3 --steps 600 --resolution 32 --device cuda --r11
```
