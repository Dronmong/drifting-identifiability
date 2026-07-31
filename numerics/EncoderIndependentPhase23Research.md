# Encoder-Independent Kernel Drifting — research pass on Phase 22

> **§2's mechanism and §4/§5's plan are SUPERSEDED by
> `EncoderIndependentPhase24Results.md`.** The Phase-24 preflight measured both
> load-bearing claims and refuted both: a code space is *less* selective than
> pixel space (realized ESS 0.944 vs 0.681), and a decoded code average is
> *worse* than a pixel average (precision 0.352 vs 0.795). Ambient dimension
> was never the obstacle — the same calibration hits its target on real-vs-real
> data in both spaces. The real driver is that the generator's **cloud** is
> concentrated (CV 0.08), which flattens the weights whatever the space.
> §1 (the two corrections) and §3 (the precision/recall diagnosis) stand, and
> §3 was confirmed. Retained unedited below because it was acted on.

## The teacher is always a blur of ≥18 images. That is the ceiling, and it is not a tuning problem.

*Research pass. No new training. Evidence: `phase22.json`, `phase8_probe.json`,
`phase20_probe.json`, `phase21_probe.json`, plus the six Phase-22 sample grids.*

---

## 1. Two corrections to the record first

**The R11 inflation hypothesis is refuted by data that was already in the
artifact.** I proposed last night that flat kernels force R11 to inflate a
blurry target harder, manufacturing the garish artifacts. Every Phase-22 arm
lands at a second-moment ratio of 0.97–1.00, so R11 acts identically in all of
them and cannot be what separates them. `EncoderIndependentPhase22Results.md`
§2 now carries the refutation next to the claim.

**`TAIL_KEEP = 32` equals `latent_dim = 32`.** The spectral-tail statistic this
program has used since Phase 10 measures variance beyond the top 32
directions, and the generator has exactly 32 latent dimensions. Nobody chose
that alignment. It means a near-linear 32-dimensional generator scores tail ≈ 0
*by construction*, so `tail` conflates "lacks high-frequency content" with
"has a near-linear latent map." Every tail number in this program carries that
confound. It does not overturn any conclusion I can identify — the Phase-22
tail ordering is far too large to be explained by it — but it needs to be on
the record and the constant should be decoupled from the latent dimension.

---

## 2. What Phase 22 actually established

Six configurations, an 18× range on the selectivity axis, 30 000 steps, four
seeds, and **not one recognizable object**. Two contrasts resolved at
p ≤ 0.003, both saying *sharpening past ESS 0.05 is catastrophic*. The best
configuration beat the baseline by 0.0205 KID (p = 0.045) and looks no more
like an image than the baseline did.

**Configuration tuning is exhausted.** That is a result, not a disappointment:
it rules out the entire family of explanations this program has been working
through for twelve phases.

### The arithmetic of the ceiling

The teacher is a kernel-weighted average of the real batch. Realized ESS
across every arm, with the resulting number of images averaged:

| arm | realized ESS | images averaged | KID |
|---|---:|---:|---:|
| A_control | 0.815 | 52.2 | +0.16084 |
| B_sharp | 0.536 | 34.3 | +0.14660 |
| **C_sharper** | 0.279 | **17.8** | +0.25988 |
| D_pos | 0.524 | 134.2 | +0.13634 |
| E_sharper_pos | 0.237 | 60.6 | +0.26413 |
| F_mix | 0.707 | 45.2 | +0.13116 |

**You cannot produce a sharp natural image as a weighted average of 18 CIFAR
images**, and 18 is the *fewest* this program has ever achieved. Below that the
cliff fires: `sharper_at_pos64` +0.113 (p = 0.0033) and `sharper_at_pos256`
+0.128 (p = 0.0001), and quadrupling the attractor set does not move it
(p = 0.20).

So the mechanism is pinned between two walls. Average many images and the
target is mush; average few and the process degenerates. Neither wall is a
hyperparameter.

### Why the walls are there

The field is a kernel estimate in **3072 dimensions from 64–256 samples**. In
that regime kernel weights cannot be selective — the pairwise distances
concentrate, so every positive looks roughly equidistant from every cloud
point. Forcing selectivity by shrinking the bandwidth does not recover
information that the sample never contained; it just reduces the effective
sample size until the estimate is noise. That is the cliff.

This is the documented failure mode of kernel particle methods in high
dimension (the SVGD variance-collapse literature) and of fixed-kernel MMD
generators in pixel space (GMMN, and the analysis in *Demystifying MMD GANs*).
**Phase 22 replicates a known negative result.** Our contribution is that we
measured where the walls are and showed the interior optimum is real.

---

## 3. The metric inversion, diagnosed

`C_sharper` scores worst on everything and looks most like photographs. The
second moments say this is not a scale effect, so it is a **rank** effect:

- real CIFAR: tail 0.1962 — ~80% of variance in the top 32 PCs
- `A_control`: tail 0.1611 — closest to real
- `C_sharper`: tail 0.0015 — 99.85% inside 32 directions

`C_sharper` is a **low-rank family of individually plausible blurry images**;
`A_control` **spans more of the space with wrong content**. High precision /
low recall versus low precision / high recall. KID and FID are
distribution-level moment matches that reward coverage, so they rank A and F
above C.

Corroboration from `phase8_probe.json`: turning R11 on moves tail from 0.0032
to 0.0474 at fixed everything else — **R11 buys rank, i.e. recall**. It was
validated on ED² and the second moment, both of which reward exactly that.
Nothing in its validation history could have detected a precision cost.

**This program has never computed precision or recall.** A coverage/quality
trade-off has therefore been able to sit behind every configuration decision
made in 22 phases, invisible.

---

## 4. What would actually produce better results

Not a better kernel, a better *space*. The ceiling in §2 is a statement about
averaging in **pixel** space: the mean of 34 CIFAR images is mush because the
pixel-space average of points on the image manifold leaves the manifold.

The same average taken in a **learned latent space** and then decoded does not,
because the decoder maps latent points back onto the manifold. That is the
standard fix in this literature — GMMN fails in data space and works in
autoencoder code space — and it is also what the paper actually does: it
drifts *inside* an encoder's latent space, which this harness has never done.

### This advances the encoder-independence target rather than abandoning it

The distinction that matters:

- **External pretrained semantic encoder** (ImageNet/DINO): the paper's
  dependency, requires outside data and supervision. This is what the program
  set out to remove, and Phases 17–18 showed it is actively *harmful* as a
  pixel-space kernel.
- **Autoencoder trained on the target distribution only**: self-contained, no
  external data, no labels, no semantic supervision. Reconstruction on the
  same 40k CIFAR images the generator already sees.

Drifting in a self-trained autoencoder latent space is still encoder-*independent*
in the sense this program cares about. And it is the first configuration in
which the paper's own ablation becomes testable: with a decoder in place we can
drift inside a self-trained latent space, a pretrained encoder's, and an
untrained one's, and compare — the experiment Phases 17–18 could only
approximate by using features as kernel weights over pixels.

### Dimension arithmetic

A 4×4×16 latent is 256 dimensions against 3072 in pixel space — a 12×
reduction, moving a 256-particle cloud from hopeless to marginal on the
particle-count-versus-dimension axis that §2 identifies as the cause of both
walls.

---

## 5. Concrete plan

**Build (this session):**

1. **Precision/recall** (Kynkäänniemi et al. 2019) on Inception features,
   k-NN manifold estimate. Without it we cannot see the trade-off in §3, and
   every future decision repeats the mistake.
2. **Power-spectrum slope** against real data — cheap, interpretable, and it
   discriminates "garish texture" from "blurry photograph" directly, which no
   current statistic does.
3. **Convolutional autoencoder** on the CIFAR train split (40k, disjoint from
   eval as always), latent 4×4×16 = 256.
4. **Latent-space drifting**: same field, same kernel, same R11, computed in
   latent space; decode only to evaluate.

**Measure the ceiling before running anything.** Score
`decode(encode(real_eval))` on KID, FID2048 and precision/recall. Latent
drifting cannot beat its own autoencoder, so that number bounds the whole
approach and must be reported alongside every result. If AE reconstruction is
itself at KID 0.10, the approach is capped near there and we should know that
in an hour rather than after an overnight run.

**Then Phase 24:** pixel-space best (`F_mix`) versus latent-space drifting at
matched budget, **≥ 8 seeds** per this program's standing rule, with KID,
FID2048, precision/recall, tail and grids for every arm.

**Declared in advance:** if latent drifting produces recognizable objects, the
program has a working method and the encoder ladder becomes meaningful. If it
produces the same mush at 256 dimensions as at 3072, then the dimension
argument in §4 is wrong, the problem is the drift objective itself rather than
the space, and the honest output is the negative result plus the mechanism
map — which is already written.

---

## 6. What I am not proposing, and why

- **Bigger generator** — Phase 8 found 36× width did nothing (on ED², so weak
  evidence), and §2 locates the bottleneck in the field's information content,
  not the generator's expressiveness. Cheap enough to carry as a control, not
  as a hypothesis.
- **Log-space bi-softmax stabilization** — built for opening the sharp regime,
  which Phase 22 measured as *worse* (p = 0.0001). Deprioritized on evidence.
- **More bandwidth/selectivity sweeps** — §2 is the argument that this family
  is exhausted.
- **Branch A (the spectral anchor)** — still untested since Phase 2 and still
  the only mechanism addressing source correctness rather than geometry. It
  belongs in the queue, but it changes *why* the field is right rather than
  *whether* the teacher is a usable image, and §2 says the second is what is
  binding.
