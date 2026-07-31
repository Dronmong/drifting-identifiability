# Encoder-Independent Kernel Drifting — Phase 14 plan

## Test the thesis, not the method

*Planning document. No results. Written after the GPU enablement
(`device_check.json`) and after confirming two capabilities the program has
never had.*

---

## 1. What just changed

Two constraints shaped thirteen phases, and both are now gone.

**Pretrained encoders are reachable.** `reference_encoder.py` opens with:
*"This repository has no pretrained image encoder and no network access
during runs, so the paper's arm cannot be reproduced here."* Verified today:
`resnet18(weights=IMAGENET1K_V1)` downloads and constructs. The A8 arm no
longer has to be a locally-trained autoencoder stand-in.

**Real FID is feasible.** `inception_v3(weights=IMAGENET1K_V1)` runs on the
GPU and yields 2048-dimensional pool features. The program has scored
everything with ED² and a normalized composite; the paper's own metric is
now available.

**And the GPU makes the scale affordable** — 7.4×/8.7×/11.6× at field cloud
64/256/512, arithmetic matching CPU to 9.4e-7. What would have been ~16 CPU
hours is ~2.

---

## 2. Why the top priority is a validity check, not the comparison

The obvious next move is "run encoder-free against encoder-based." It is the
right *second* move.

The program's own data raise a prior question. In the Phase-1 screen — the
only encoder-vs-encoder-free measurement that exists — **raw pixels beat the
learned-geometry arm by ~5×** (ED² 0.186 against 0.900), and every fixed
compositional family did worse than raw pixels, three times over. Meanwhile
the paper's central ablation reports FID tracking encoder quality closely,
down to 3.36 for a classification-fine-tuned encoder.

Those two facts point opposite ways. Either

- **(a)** at this scale, semantic geometry genuinely does not help — a real
  and publishable finding that would make encoder-independence easy; or
- **(b)** the harness cannot see what the encoder contributes, in which case
  every encoder-free result in thirteen phases is uninformative about the
  thesis.

**Nothing in the program distinguishes (a) from (b).** Until it does, running
the comparison would produce a number nobody could interpret. So Phase 14A is
a *precondition*, and it is cheap.

---

## 3. Phase 14A — the encoder-quality ladder

**Question:** does this harness reproduce the paper's qualitative finding
that generation quality tracks encoder quality?

One recipe, geometry varied, everything else fixed:

| arm | geometry | what it isolates |
|---|---|---|
| G1 | **pretrained ResNet18** features (ImageNet) | the paper's regime |
| G2 | **randomly-initialized ResNet18**, same architecture | pretraining vs architecture |
| G3 | the local autoencoder stand-in (Phase 1's A8) | continuity with the old record |
| G4 | **raw pixels** | encoder-free |
| G5 | deliberately degraded (heavy blur / few random projections) | the bad-encoder end |

**Non-negotiable design constraints**, each learned the hard way:

- **Every geometry gets its own target-ESS-0.9 bandwidth calibration.**
  Phase 7 showed bandwidth is worth 4.9×, and Phases 2–6 compared geometries
  at an uncalibrated bandwidth. Repeating that would invalidate the whole
  comparison.
- **R15 admissibility first.** A geometry whose kernel is numerically dead is
  not evidence about geometry; report its health numbers and do not score it.
- **Feature normalization must be declared per geometry.** The paper's
  temperature grid is calibrated for *normalized encoder features*, which is
  exactly why τ ∈ {0.02, 0.05, 0.2} collapsed the kernel on raw pixels
  (93.8% dead rows). With real encoder features **the paper's declared
  operating point becomes testable for the first time** — worth running as a
  sub-arm on G1.
- **Both metrics**: ED² (comparable with all thirteen phases) and **FID**
  (comparable with the paper). Divergence between them is itself a finding.

Scale: CIFAR-10 at **32×32**, 3 seeds, matched budget, on GPU.

### The declared readings

- **G1 > G2 ≈ G3 > G4 > G5** with a clear gradient → the harness sees encoder
  quality, the paper's ablation reproduces, and 14B's comparison is
  meaningful.
- **G4 (raw) at or near the top** → the harness contradicts the paper. Before
  claiming (a), check the sub-arm at the paper's operating point and the FID
  metric; if raw still wins under the paper's own configuration and metric,
  that is the program's headline result and it is a strong one.
- **No ordering at all** (everything within noise) → the harness is blind
  (b). Then the priority becomes fixing the measurement, and **no
  encoder-free claim can be made from any existing phase.**

**G2 is the highest-information arm.** If a randomly-initialized ResNet
matches a pretrained one, the "semantic encoder" story collapses and
encoder-independence is nearly free. If pretrained beats random decisively,
we have quantified exactly what must be replaced — which is what Branch B
tried and failed to do blind.

---

## 4. Phase 14B — the comparison, if 14A validates

Consolidated encoder-free against encoder-based, both carrying every reform
the program has earned:

| arm | geometry | reforms |
|---|---|---|
| F0 | raw pixels | none *(the historical baseline)* |
| F1 | raw pixels | ESS-0.9 + R11 |
| **F2** | **raw pixels** | **ESS-0.9 + R11 + anchor** |
| F3 | pretrained ResNet18 | ESS-0.9 |
| F4 | pretrained ResNet18 | ESS-0.9 + R11 |

**F2 has never been assembled.** The ESS-0.9 bandwidth was found in Phase 7
and no arm has used it; the anchor has been disabled since Phase 2. It is the
best encoder-free configuration this program can build and it has never been
run.

**The deliverable is one number**: `FID(F2) − FID(F4)`, the price of dropping
the encoder, with a confidence interval over seeds. That number does not
exist today and is the thing the whole program was for.

---

## 5. Phase 14C — the anchor's real job

The anchor is the only *theoretical* argument for encoder-freedom: it is what
makes `loss = 0 ⟹ p = q` when the geometry kernel is not
measure-determining. Phase 0 measured that the fixed geometries **are** blind
to real collisions (the wavelet branch cannot see `color_swap`, p = 0.74).

So the anchor is not a 3.5% accessory — it is the thing that makes an
encoder-free kernel *sound* rather than merely adequate. 14C runs the
collision suite against G1–G5: **which geometries are blind, and does the
anchor close their blindness?** A pretrained encoder that is itself blind to
a collision would be a notable result about the paper's method, not just
about ours.

---

## 6. What is deprioritized, and why

- **The Phase-13 settling run** (visits-matched pair-bank sweep). It is now
  cheap on GPU and it would close the amortization question, but it serves
  the *method*, not the thesis. Run it only if 14A/14B leave GPU time idle.
- **Further mechanism work.** Thirteen phases, fourteen refuted hypotheses,
  one shape law. It is genuinely interesting and it is answering "why does
  stop-gradient regression contract" — a question about drifting in general.
  **Stop.**
- **Reviving Branch B** (fixed compositional geometry). Three negatives.
  14A's G2 arm will say more about whether a *learned* geometry is even
  necessary than another wavelet family would.

---

## 7. Risks, and what would invalidate this

- **CIFAR-32 is still not the paper's regime.** The paper reports ImageNet
  FID. A CIFAR-32 result bounds the question; it does not settle it, and the
  write-up must say so.
- **ResNet18-ImageNet is not the paper's encoder.** The paper uses
  self-supervised encoders (DINO-family). ResNet18 is a *real* pretrained
  semantic encoder, which is a large step up from an autoencoder stand-in,
  but the substitution must be stated wherever G1 is quoted.
- **FID at 32×32 through a 299×299 Inception is noisy** and conventionally
  computed with ≥10k samples. With 512–2048 samples it is indicative, not
  publishable; report the sample count beside every FID.
- **The GPU changes numerics.** Arithmetic matches CPU to 9.4e-7, but Phase
  14 should be run **entirely on GPU** so it is internally consistent, and
  its numbers should not be spliced into CPU-era tables without saying so.

---

## 8. Recommended order

1. **14A ladder** (~45 min GPU) — the precondition. Nothing else is
   interpretable without it.
2. **14B comparison** (~1 h GPU) — the number the program exists to produce.
3. **14C blindness/anchor** (~30 min) — whether encoder-free is *sound*, not
   just competitive.

Total ≈ 2¼ hours of GPU time to convert thirteen phases of method work into
an actual answer about encoder independence.

**One caveat worth stating plainly before starting:** 14A can return a result
that invalidates the interpretation of a lot of earlier work. If the harness
turns out to be blind to encoder quality, then the encoder-free results are
not evidence for the thesis — they are evidence that the measurement cannot
see the difference. That is a real possibility given the Phase-1 inversion,
and it is better to find out in 45 minutes than after another thirteen
phases.
