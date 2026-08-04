# Anchored self-feature drifting — mechanism and build specification

**Working name:** ASFD
**Status:** implementation specification; no experiment authorized
**Date:** 2026-08-03

**Supersedes** the mechanism and procedure sections of
[`AnchoredSelfFeatureDriftingResearchPlan.md`](AnchoredSelfFeatureDriftingResearchPlan.md).
That document remains the record of the literature survey, the novelty
assessment, and the reasoning that produced this design. Where the two
disagree, this document governs; the reasons are recorded in
[`AnchoredSelfFeatureDriftingMechanismAudit.md`](AnchoredSelfFeatureDriftingMechanismAudit.md).

---

## 1. What is being built

A one-call image generator that learns its own training geometry.

We train a raw-pixel, encoder-free, one-network-call generator (CAP-EMF-1) on a
single CIFAR-10 class. We then freeze a copy of its own trunk and use that
frozen copy's hidden states as a *training-only* similarity geometry for a
distributional drift correction applied to a continuation of the same
generator.

Correctness never routes through the learned geometry. A separately squared
raw-pixel Laplace drift energy runs alongside it with a strictly positive
coefficient, and it is that term — not the feature term — that carries the
project's formal identifiability result.

**The complete inference object is one forward pass of one network.** No
encoder, no VAE, no diffusion teacher, no feature network, no reference bank,
and no iterative solver appear at sampling time.

### 1.1 The claim this is built to support

> On a preregistered single-class 32×32 pixel task, a one-call generator used
> its own frozen foundation features as a training-only drifting geometry.
> Adding this self-feature field to independent raw identifying and spectral
> anchors improved the matched held-out fidelity/coverage tradeoff without an
> external encoder, VAE, or diffusion teacher.

Everything this may **not** claim is in §12.

### 1.2 What distinguishes it

The closest published work is Teacher-Feature Drifting (arXiv 2605.07327),
which removes the drifting paper's separate encoder by using a **pretrained
diffusion teacher's** intermediate features. The distinction that remains:

| Teacher-Feature Drifting | ASFD |
|---|---|
| starts from a pretrained multi-step diffusion teacher | trains a one-call foundation from target data |
| distills the teacher and inherits its geometry | uses a frozen copy of the generator's own foundation |
| no extra encoder, but a powerful teacher is required | no external teacher or separately pretrained representation |
| feature geometry carries the objective | an independent raw field carries the exact source-space anchor |

The open question is whether **self-founded**, rather than teacher-distilled,
feature drifting can improve a one-call generator without making correctness
depend on the learned feature map.

---

## 2. Preconditions

ASFD does not begin until all of the following hold. Each is a hard gate.

| # | precondition | evidence required |
|---|---|---|
| P1 | CAP-EMF-1 foundation passes its capability gate | recognizable uncurated automobiles, nontrivial train-only classifier recognizability, no memorization veto, adequate raw variance and high-frequency retention, stable one-call forward counter, mature EMA |
| P2 | B2.5 units 501 and 502 complete and aggregate | resolves whether the spectral anchor and raw drift term compound — ASFD stacks a third term on exactly that question, and unit 500 fails two of four preregistered conditions |
| P3 | Feature qualification gate passes (§7) | run on target-training images only |
| P4 | Preflight passes (§8, Stage C) | mechanical, numerical, and cost |

**A failed foundation cancels ASFD outright.** Hidden states of a generator that
cannot produce recognizable images are not a credible semantic metric, and no
correction manufactures spatial structure the endpoint model never learned.

---

## 3. The mechanism

### 3.1 Notation

| symbol | meaning |
|---|---|
| `f_θ` | the online one-call generator being trained |
| `θ*` | frozen EMA weights at the foundation checkpoint |
| `h_{θ*}` | frozen copy of the trunk, used only for feature extraction |
| `p` | the target law on raw pixels |
| `q_θ` | the law of `f_θ(latent)` |
| `j ∈ J` | feature level (four; §4.2) |
| `ℓ ∈ L` | descriptor location within a level (66; §4.3) |
| `r ∈ R` | bandwidth radius (three; §5.3) |
| `z` | a probe point |
| `τ` | Laplace bandwidth |

Roles are three-way split and never shared: **probes** are noised target
samples, **positives** are fresh target samples, **negatives** are a
differentiable batch from the online generator.

### 3.2 The objective

The total training loss at a correction event is

```
L  =  L_EMF  +  λ₁ · L_B1  +  λ_raw · E_raw  +  λ_self · E_self ,     λ_raw > 0
```

where `L_EMF` is the foundation's direct-`x` Euler Mean Flow objective, `L_B1`
is the encoder-free spectral anchor, and the two energies are defined below.

**This sum is descended as written.** The applied update is exactly `∇L`; no
component is projected, reweighted by alignment, or otherwise modified. Only
the ordinary global gradient clip already used by the foundation is applied
afterwards, to the summed gradient. See §6.1 for why this is not negotiable.

### 3.3 The normalized Laplace mean-shift field

Both energies are built from one object. For a bandwidth `τ`, a probe `z`,
positives `{y⁺_a}` and negatives `{y⁻_b}` living in a common space,

```
              Σ_a  e^{−‖z−y⁺_a‖/τ} y⁺_a        Σ_b  e^{−‖z−y⁻_b‖/τ} y⁻_b
  V_τ(z)  =  ───────────────────────────  −  ───────────────────────────
              Σ_a  e^{−‖z−y⁺_a‖/τ}            Σ_b  e^{−‖z−y⁻_b‖/τ}
```

— a difference of two normalized barycenters. The displacement `−z` cancels
between them. Implemented as `softmax(−distance/τ)`, which analytically equals
the normalized Laplace weights while subtracting the row maximum internally, so
a remote-but-finite row cannot become an all-zero row.

Positives and probes are detached. **Negatives are not**: the graph must run
from `E` through `y⁻` and, in the feature case, through the frozen trunk's
input Jacobian into `θ`.

### 3.4 The raw term — this is the one that carries correctness

```
  E_raw  =  mean_{r ∈ R}  mean_z  ‖ V^raw_{τ_r}(z) ‖²
```

computed on flattened raw pixels, with probes drawn as target samples plus
Gaussian noise of standard deviation `0.05`.

**Why it identifies.**
[`laplaceZeroDrift_identifies_euclidean`](../DriftingIdentifiability/LaplaceEuclideanConverse.lean#L29)
proves that pointwise-zero normalized Laplace mean-shift drift forces `p = q`
for arbitrary probability measures in every finite dimension, with no
hypotheses beyond `τ > 0`. `ZeroDrift` in that statement is genuinely
pointwise, not almost-everywhere — the repository keeps a separate `ZeroDriftAE`
and the distinction is tracked.

The probe law is a Gaussian convolution of a compactly supported measure and
therefore has **full support on ℝⁿ**. So zero integrated energy plus continuity
of the field gives pointwise zero on all of ℝⁿ, and the converse applies:

```
  L_corr = 0   ⟹   E_raw = 0   ⟹   p = q          (λ_raw > 0)
```

No injectivity assumption on the self-feature map appears anywhere in this
implication. Multi-radius does not weaken it — `E_raw = 0` requires every
radius's field to vanish, and any single one suffices.

**This is an exact population statement.** It is not a claim that a finite
minibatch loss identifies a law, nor that a nonzero trained loss says anything
at all. §12 records this.

### 3.5 The self-feature term

```
  E_self  =  mean_{j, r, ℓ, z}   (1/C_j) · ‖ V^{j,r}_ℓ(z) ‖²
```

where `V^{j,r}_ℓ` is the field of §3.3 computed **inside** feature level `j` at
descriptor location `ℓ` with bandwidth `τ_{j,r}`, over normalized features
`h̃_j` (§5.2). Locations vectorize as a leading batch dimension.

This term has **no identifiability guarantee and is not asked to have one**.
The feature pushforward of the probe law does not have full support in feature
space, so the energy→pointwise bridge does not transfer. Its job is purely
operational: to supply image-aware finite-sample neighborhoods that raw pixel
distance does not.

### 3.6 Separate squares, on every index

Fields from different levels, radii, locations, and probes are **squared before
they are averaged**. Squaring an aggregate would permit two wrong fields to
cancel and would destroy the zero-set interpretation: the zero set of a sum of
squares is the intersection of the zero sets. This is enforced on `j`, `r`, `ℓ`
and `z` alike, and a regression test constructs two canceling synthetic fields
to prove the implementation does not collapse them (§9.2).

### 3.7 The three independence notions

The design must not conflate these:

1. **E1 — no external encoder.** No DINO, CLIP, MoCo, SimCLR, MAE, VAE encoder,
   or diffusion teacher supplies training geometry. ✅ satisfied.
2. **E2 — no separately learned representation.** No separately optimized
   network decides which distributions are equal. ✅ satisfied — the trunk is
   learned *while building the generator*, not beside it.
3. **E3 — correctness independent of learned features.** Even if the feature map
   collapses two different images, a source-space term prevents that collision
   from becoming an exact equilibrium. ✅ satisfied by §3.4.

ASFD **does** use an internal representation and does not claim to be
representation-free. The honest statement is: *no external or separately
pretrained representation is needed, and exact population correctness does not
rely on the self-feature map being injective.*

---

## 4. Architecture

### 4.1 The foundation trunk

CAP-EMF-1, frozen before ASFD begins:

| component | value |
|---|---|
| data | 5 000 CIFAR-10 automobile train images; 1 000 test images sealed |
| representation | raw RGB pixels; no VAE, encoder, teacher, or reference bank |
| inference | one direct endpoint call |
| objective | audited direct-`x` Euler Mean Flow, JVP-free |
| image tokens | patch 2 → **256 tokens (16×16)** |
| trunk | width 512, depth 12, 8 heads, long U-shaped skips |
| conditioning | AdaLN-Zero-style time/interval |
| output | shallow convolutional pixel head |
| optimizer | AdamW, `1e-4`, betas `(0.9, 0.95)`, zero weight decay |
| batch | 64 effective |
| training | 160 000 updates |
| EMA | 0.9999 |

**The trunk is a U-ViT with long skips and no spatial downsampling.** Encoder is
blocks 1–6, decoder is blocks 7–12, each decoder block fusing a reversed encoder
skip
([`stage_pmf_r/model.py:43–50, 123–130`](encoder_independent_drifting/stage_pmf_r/model.py#L43)).
Every block runs at the same width and the same token count.

**There is no bottleneck.** The term is not used in this specification; block 6
has the same shape as block 1, and importing the convolutional-U-Net intuition
that a mid-network stage is compressed and abstract would be wrong here.

### 4.2 Feature taps

Four levels, spanning the trunk:

| label | tap point |
|---|---|
| `enc_mid` | output of encoder block 3 |
| `enc_final` | output of encoder block 6 |
| `dec_mid` | output of decoder block 9, **after** its long-skip fusion |
| `dec_final` | output of decoder block 12, **before** `final_norm` and `pixel_head` |

Rationale: TFD's ablation (Fig. 4b) finds that combining intermediate and deep
representations matters, that deep-only is worse, and that five spread levels
beat three. A *distributional* field over many levels does not over-constrain
the way a per-sample perceptual regression does — each level contributes an
independent nonnegative discrepancy whose zero set contains the true one, so
extra levels tighten an intersection rather than over-determining a point. This
is the same argument as §3.6.

`dec_final` is taken **before** the norm and pixel head so the descriptor is not
a near-affine image of output pixels, which would make it redundant with §3.4
by construction.

Levels that fail the inter-level redundancy check (§7.3) are **dropped, not
tuned**.

### 4.3 Descriptors

Per level, from the 16×16 image-token grid:

- **64 local vectors** — non-overlapping 2×2 average pool to 8×8;
- **1 global channel-mean vector**;
- **1 global channel-standard-deviation vector**.

**66 vectors of width `C_j = 512` per level.** Local structure is retained
rather than collapsing to a single pooled vector.

**Token-grid extraction is conditioning-dependent.** The trunk as currently
implemented prepends two conditioning tokens
(`torch.cat((time_token, interval_token, patches), dim=1)`,
[`model.py:121`](encoder_independent_drifting/stage_pmf_r/model.py#L121)), so the
image grid is `tokens[:, 2:]`; under CAP-EMF-1's AdaLN-Zero conditioning there
are exactly 256 tokens and no slice. **Write extraction against whichever
CAP-EMF-1 freezes, and assert it** — the forward-parity test cannot catch an
off-by-two slice, because ordinary outputs are genuinely unchanged by a
read-only hook.

**A known limitation, to be measured not assumed:** local fields are
position-locked — location `ℓ` of a generated image is only ever compared with
location `ℓ` of a target image — so 64 of 66 vectors (97% of `E_self`) are
sensitive to pose and translation. Under the horizontal flip recorded in the
augmentation bits, token `(i,j)` maps to `(i, 15−j)`. §6.4 requires the
global-versus-local energy split to be logged so this is visible.

### 4.4 Feature extraction path

Features are **never** extracted from clean images. For each image and each
feature draw:

```
  x_{t_f} = (1 − t_f)·x + t_f·ξ ,      ξ ~ N(0, I)
  φ_j(x; ξ) = h^{(j)}_{θ*}( x_{t_f},  t = t_f,  interval = 0 )
```

This is the trunk's **native** corruption path — a time-conditioned model
evaluated in-distribution, not an arbitrary additive perturbation. CleanDIFT
establishes that diffusion features are poor on clean inputs because training
occurred on noisy states, and that averaging arbitrary noise draws does not
repair a mismatched extraction regime.

`t_f` is **selected, not assumed** (§5.4). Probe, positive, and negative roles
receive independent noise streams; a positive and a negative are never paired
through the same noise tensor.

### 4.5 Freezing discipline

`h_{θ*}` is a separate module with `requires_grad = False` on all parameters,
in evaluation mode, with a recorded source-checkpoint hash and frozen
normalization statistics. It is never updated.

> **Freezing parameters must not detach inputs.** The generated branch needs
> `∂h_{θ*}(x)/∂x` for `E_self` to reach `θ` at all. Implementing the frozen
> forward under `torch.no_grad()` yields a silently dead branch that still
> trains, still logs a decreasing loss, and still produces an artifact.

The online generator remains the only exported inference object. The frozen
module, banks, kernels, B1, and raw probes are all training-only.

---

## 5. Calibration — every constant is derived, none inherited

### 5.1 Nothing transfers from B1 or B2

| constant | frozen value | why it cannot be reused |
|---|---:|---|
| B2 bandwidth τ | 7.085388360479058 | calibrated on all-class CIFAR-32 raw pixels, median pairwise distance 37.49; automobile-only is a tighter cloud |
| B2 event weight λ | 1.9294302093274076e-4 | calibrated against the **F3B bridge's** flow loss; EMF's loss has an unrelated gradient scale |
| B1 event weight | 0.9310125645774651 | same reason |
| B1 projected scale | 0.4299860893300136 | derived from bridge-scale activations |

What transfers is the **form** of each term and the **calibration procedure**.
A test asserts that no B1 or B2 freeze artifact is loaded anywhere in the ASFD
path (§9.2).

Consequence: the raw arm is a **new development configuration**, not a
continuation of a confirmed one. §12 records this.

### 5.2 Feature normalization — two stages, both target-only, both frozen

**Stage 1, conditional per-channel.** On the target-only calibration allocation,
compute the token-level PC1 variance share of level `j`. If `PC1 > 0.35`, apply
frozen per-channel scales

```
  s_{j,c} = max( σ_{j,c} ,  0.1 · median_c σ_{j,c} )
```

The floor bounds amplification of low-variance noise channels at 10×. If
`PC1 ≤ 0.35`, this stage is the identity. Record which branch fired.

This stage exists because transformer hidden states routinely have a few outlier
channels dominating the L2 norm, in which case a Laplace distance silently
reduces to a one- or two-dimensional statistic.

**Stage 2, scalar level scale.** Computed after stage 1:

```
  S_j = (1/√C_j) · mean_{a≠b, ℓ}  ‖ h_j(x_a)_ℓ − h_j(x_b)_ℓ ‖₂
```

Use `h̃_j = h_j / S_j` and freeze `S_j`. All locations within a level share one
scale. Reject zero, non-finite, or poorly resolved scales.

Both stages are frozen before any correction event, so the metric cannot move in
response to model failure.

### 5.3 Bandwidths — three radii spanning a real range

```
  R = { 0.10 , 0.35 , 0.85 }        target median off-diagonal ESS fraction
```

Calibrated **per feature level** (and separately for the raw branch) by
bisection on `τ`, using self-excluded effective sample size exactly as
[`stage_b2/core.py:244`](encoder_independent_drifting/stage_b2/core.py#L244)
does. Target-only data.

This spans ~8.5× in ESS. The alternative considered and rejected — three broad
radii — spans under 2× in `τ` and contains no local regime, so all three fields
would be blind to sub-bandwidth structure in the same way and averaging them
would not repair it. TFD's ablated radius set spans 10×.

**Health limits, applied at calibration time on the target side:**

| statistic | limit |
|---|---|
| 5th-percentile ESS | `≥ 0.10` |
| 95th-percentile maximum weight | `≤ 0.50` |

**Fallback ladder.** If the smallest radius fails these at the frozen batch
size, step it along `{0.10, 0.15, 0.20}` and record which rung was used. Never
widen it silently.

**The raw branch is multi-radius too.** The rank collapse measured in B2 is a
property of broad-bandwidth barycenter matching, not of pixel geometry; a single
raw radius at ESS 0.60 reproduces it.

### 5.4 `t_f` — selected by target-only audit

Run the §7 qualification audit over

```
  t_f ∈ { 0.05 , 0.10 , 0.20 , 0.35 , 0.50 }
```

on target-training images only, and select the value maximizing a predeclared
composite of benign-vs-random AUC, the per-band sensitivity profile (§7.2), and
inter-level non-redundancy (§7.3), subject to every hard threshold passing.
Freeze one value; record the full profile across the grid.

This is calibration, not tuning — the same status as bandwidth calibration, on
no held-out data. It does not create the hazard of varying `t_f`, layer set, and
kernel together *inside* the training experiment, which remains forbidden.

**The tradeoff being navigated.** Along `t_f`, injectivity and semantic
abstraction pull in opposite directions: at small `t_f` a direct-`x` trunk's task
is near-identity, so hidden states are near-invertible codes — maximal
injectivity, minimal abstraction; at large `t_f` the trunk must use global
context — more abstraction, more collision risk. ASFD wants abstraction from the
feature branch and relies on the raw branch for correctness, so it can afford to
sit further toward abstraction than an injectivity-seeking design would.

Reference point: TFD ablates `σ_tf ∈ {0.02, 0.1, 0.5}` and selects **0.1**, with
0.02 converging slower to a worse FID and 0.5 also degrading. Up to the
`(1−t_f)` signal scaling — required here because the trunk is time-conditioned —
that matches `t_f = 0.10`. It is a reference point, not a substitute for
measuring it on this trunk.

### 5.5 Coefficients

`λ₁`, `λ_raw`, `λ_self` are calibrated on **outcome-blind** training-only
events so that the per-component ratios of §6.2 hold at calibration time. They
are frozen before the development fork and recorded in the artifact.

### 5.6 Batch roles

| role | size | cost |
|---|---:|---|
| positives | **256** | cached; storage only |
| probes | 64 | cached |
| negatives | **64** | one generator forward each |

Asymmetric deliberately. At 64-vs-64 the real-versus-real sampling floor is
**54–67% of total measured energy** in this repository's artifacts, so most of
what the correction differentiates is sampling noise. The floor falls roughly as
`1/n`; 64 → 256 positives should lift signal-to-floor from ≈ 0.85 toward ≈ 3.
Because positives are precomputed, this is nearly free.

### 5.7 Target feature banks

After freezing `θ*`, precompute **4 independently noised views** per training
image for the positive bank and 4 disjoint views for the probe bank.

Storage: `66 × 512 × 4 levels × 4 views = 540 672` values/image ≈ **1.08 MB in
fp16**; × 5 000 images ≈ **5.4 GB**. Host-RAM resident, pinned, paged to the GPU
at ≈ 69 MB per event. Store fp16; accumulate kernels and energies in fp32.

Four views rather than two because the positive barycenter built from a small
fixed view set is a fixed random function whose deviation from the population
barycenter is a bias the generator can partly learn to match. **Log the
train-bank minus fresh-bank energy gap at every checkpoint** — that gap is the
direct measurement of this bias.

Record image indices, augmentation bits, feature-noise seeds, feature
configuration, teacher hash, dtype, normalization scales, and bank hashes.
Evaluation uses a separately seeded fresh bank.

The generated branch is always recomputed with fresh feature noise and remains
differentiable.

---

## 6. Training integration

### 6.1 The summed gradient

At a correction event, compute `g₀ = ∇L_EMF`, `g₁ = ∇L_B1`, `g_r = ∇E_raw`,
`g_s = ∇E_self`, and apply

```
  g_total  =  g₀  +  λ₁g₁  +  λ_raw·g_r  +  λ_self·g_s
```

followed by the foundation's ordinary global gradient clip.

**No projection.** An earlier draft of this design projected each auxiliary
component so it could not oppose `g₀` to first order. That is rejected for two
reasons:

1. A projected update is generally **non-conservative** — there is no potential
   whose stationary points the dynamics seek — so §3.4's implication would apply
   to a loss the optimizer never descends, and `λ_raw > 0` would do no work.
2. The raw anchor is redundant precisely when it agrees with `g₀`; it does work
   only when it disagrees. Projection deletes the disagreeing component and
   keeps the agreeing one, making the term loudest in the logs exactly when it
   is doing least.

B1, B2, and B2.5 all use plain weighted sums for the same reason. Gradient
linearity means components may be backpropagated sequentially — to avoid
retaining two trajectory graphs — before one clip and one optimizer step; that
is still `∇L`.

### 6.2 Per-component caps

| component | cap on `‖λᵢgᵢ‖ / ‖g₀‖` |
|---|---:|
| B1 spectral | 0.15 |
| raw Laplace | 0.10 |
| self-feature | 0.10 |

Each cap applies to its own weighted gradient norm, **independently**, after AMP
unscaling. The total auxiliary norm reaches 0.35 in the ASFD arm and 0.25 in the
raw arm.

**This asymmetry is intentional and is the correct design.** A shared total cap
would give the ASFD arm ≈ 29% less raw anchor and ≈ 29% less B1 than the arm it
is compared against, confounding "added a semantic term" with "cut the
protection by a third" — and the confound would point the wrong way, since the
arm carrying the new pressure would be the one with weakened protection. B2.5
settled this principle: the same treatment level must appear in the single and
combined cells, and greater total correction compute in the combined cell *is
the defined joint treatment*, not a confound. The realized total is reported as
a Pareto cost.

### 6.3 Cadence

One correction event every **10** updates, bounding the nominal cadence-averaged
influence near 2.5–3.5%.

### 6.4 Abort criteria

Outcome-based, not alignment-based. **A correction that never opposes the
primary gradient is useless**, so mild gradient opposition is the working
regime, not a fault.

| abort | threshold |
|---|---|
| non-finite gradient, broken bank hash, zero generated-input Jacobian | any occurrence |
| feature-space effective rank vs the arm's own step-0 value | `< 0.70`, two consecutive logged checkpoints |
| raw-pixel effective rank vs the paired control | `< 0.70`, two consecutive logged checkpoints |
| **raw energy below the real-versus-real floor** | any checkpoint |
| negative-side median ESS, any level × radius | `> 0.90` sustained |
| anti-parallel auxiliary, `cos(g₀, gᵢ)` | `< −0.8` over a 200-event window |

Two of these are new and evidence-driven:

**Sub-floor raw energy.** In B2.5 unit 500, B2's held-out raw energy was 13.933
against a real-versus-real floor of 14.103. A correctly distributed sample
cannot beat that floor; scoring below it means the generated cloud produces less
field discrepancy than an independent real sample, i.e. it is less variable than
real data in exactly the direction the estimator measures. That is estimator
exploitation and it is the same event as B2's 38–40% rank collapse.

**Negative-side ESS.** If it approaches 1, the negative barycenter approaches a
plain batch mean and the energy silently degenerates to first-moment matching.
B2's negative-side ESS ran 0.666 → 0.702 across audits with nothing watching it;
[`stage_b2/core.py:214`](encoder_independent_drifting/stage_b2/core.py#L214)
already records the statistic, only the threshold was missing.

Gradient cosines outside the last row are logged and interpreted, never acted
on.

### 6.5 Mandatory diagnostics

At every logged correction event and checkpoint:

- each loss and each pre-cap gradient norm;
- all pairwise gradient cosines;
- cap activation and realized **post-cap per-component ratios for every arm**,
  so the dose match of §6.2 is auditable rather than asserted;
- ordinary gradient-clipping status;
- per level and radius: ESS median / 5th percentile / minimum, maximum-weight
  median and 95th percentile, distance median and CV, field RMS, energy, and
  row-sum numerical error — **for both the positive and the negative side**;
- raw energy expressed as **excess over the real-versus-real floor**, not as a
  raw number;
- the **global-versus-local** energy split across the 2 global and 64 local
  vectors;
- the **train-bank minus fresh-bank** energy gap;
- frozen-teacher input-gradient norm;
- raw-pixel and feature-space effective rank;
- wall time, memory, model-forward counts.

---

## 7. Feature qualification gate (Stage B)

Run on **target-training images only**. No official test image is accessed. The
branch is not activated merely because a checkpoint exists.

### 7.1 Ordering — fail fast

Run **G7 and G8 first**. Both are cheap and target-only, and they test the two
assumptions the entire branch rests on: that the trunk sees the bands the
foundation is missing, and that the chosen levels carry distinct information. If
either fails, the ASFD arm is canceled before any bank construction or
coefficient calibration is spent.

### 7.2 G7 — two-sided per-band sensitivity

Inject fixed-energy perturbations into each orthonormal Haar band and measure
the feature-distance response relative to a normalized raw-pixel control:

```
  ρ_b = Δ_feature(band b) / Δ_raw(band b) ,     b ∈ {LL, LH, HL, HH}
```

**Require `ρ_b ∈ [0.25, 4.0]` in every band, for every level.**

The upper bound rejects the hypersensitivity seen in the Phases 17–18 pretrained
ResNet harness. The **lower bound is the point**: the trunk allocates capacity
under an MSE objective that underweights exactly the bands an MSE-trained
generator is weakest in, and S3R's EMF arm measured final raw Haar variance
ratios LL 0.509, LH 0.410, HL 0.512, **HH 0.159**. A feature map with zero
high-frequency sensitivity would pass a one-sided check trivially, and that is
the failure this architecture is most likely to have.

Report the full profile alongside S3R's Haar table.

### 7.3 G8 — inter-level non-redundancy

For every pair of levels, on target-only data:

- median field cosine `< 0.90` at the audit events, **and**
- linear CKA between level descriptors `< 0.95`.

Levels failing the pair test are dropped. This is what prevents the branch from
being a silent no-op through redundant taps.

### 7.4 G9 — local-token rank and concentration

Effective rank `≥ 16` and no single principal component above `50%` of variance,
applied to the **local-token** descriptor as well as the global one. If
token-level PC1 exceeds 0.35, §5.2's per-channel stage fires; if it still
exceeds 0.50 afterwards, the level fails.

### 7.5 G1–G6

| gate | requirement |
|---|---|
| G1 | benign-pair distance below random-pair distance, AUC `≥ 0.80` — computed once on the invariant global mean/std descriptor and once on local tokens **after undoing the known horizontal flip and spatial translation** |
| G2 | patch-shuffled or phase-scrambled images farther than benign variants in `≥ 80%` of paired cases |
| G3 | global feature effective rank `≥ 16`, no PC above `50%` |
| G4 | pairwise-distance coefficient of variation `≥ 0.05` |
| G5 | superseded by G7 |
| G6 | the semantic field is not numerically near the raw field: **`\|cosine\| < 0.90`** at the median audit event |

G6 is tightened from `0.995`, which permits ~99% shared variance. A branch that
is 90%-aligned with the raw branch is adding weight, not geometry.

Note the audit/deployment mismatch explicitly: G1 undoes known flips and
translations *for the audit*, but the training field gets no registration
(§4.3).

### 7.6 Mechanical checks

1. Adding feature extraction changes ordinary model outputs by at most the
   declared mixed-precision tolerance.
2. Frozen parameters receive no gradients.
3. Generated images receive finite, **nonzero** input gradients through every
   selected level.
4. Replaying hashes and noise streams reproduces cached features.
5. CPU float64 and GPU calculations agree on a tiny deterministic case.

### 7.7 On failure

**Do not tune layers on held-out image quality.** Record *which* gate failed,
because the remedies differ, then follow §11.

---

## 8. Experimental sequence

### Stage A — foundation

Run CAP-EMF-1 without any correction. Must pass P1 (§2). If it fails, ASFD
stops.

### Stage B — frozen-feature qualification

Freeze the EMA trunk, expose the four levels, run §7 over the `t_f` grid, then
build banks and calibrate scales and bandwidths (§5). No official test image is
accessed.

### Stage C — preflight

Before any long continuation:

1. feature-extraction parity, including the token-grid assertion;
2. field values on equal, shifted, and collapsed toy feature laws;
3. finite-difference the semantic energy w.r.t. generated features;
4. gradients reach online parameters **only** through generated negatives and
   the frozen input Jacobian;
5. separate-squares behavior verified with two canceling synthetic fields;
6. summed-gradient regression — the applied update equals `∇L` to tolerance;
7. per-component cap correctness on weighted norms after AMP unscale;
8. all component weights calibrated outcome-blind;
9. 500 ordinary updates and 50 correction events;
10. measured wall time and peak memory, projected to full length.

### Stage D — paired development fork

Three arms cloned from exactly the same foundation checkpoint:

| arm | continuation |
|---|---|
| `EMF-control` | EMF only |
| `EMF-raw` | EMF + B1 + raw Laplace energy |
| `EMF-ASFD` | EMF + B1 + raw Laplace energy + frozen self-feature energy |

Same flow batches, latents, EMF time samples, positive indices, update count,
optimizer state, EMA state, and evaluation cohorts. Auxiliary streams are
role-separated and deterministic. Per-component caps (§6.2) so that "raw
present" and "B1 present" mean the same thing in every arm. No test images.

**Advancement gate.** `EMF-ASFD` advances only if, relative to `EMF-raw`:

1. it reduces **fresh-bank** semantic energy;
2. its fresh-bank raw energy is not worse by more than 5% **and is not below the
   real-versus-real floor**;
3. it retains `≥ 90%` of the incumbent's **raw-pixel** effective rank;
4. it retains `≥ 90%` of the incumbent's **feature-space** effective rank;
5. it does not increase **within-class maximum pairwise SSIM**;
6. precision and recall stay within prospective uncertainty margins;
7. duplicates and nearest-training-image concentration do not increase;
8. the semantic post-cap gradient share is nonzero; and
9. projected cost is compatible with the remaining budget.

Conditions 4 and 5 exist because a within-class semantic collapse — samples
varying in colour, background and texture while repeating one car pose — can
preserve raw-pixel rank and a raw-pixel characteristic-function anchor while
being exactly the failure TFD reports its coverage term prevents.

**On coverage protection.** B1 is a random-Fourier-feature characteristic-function
criterion in **raw pixel space**
([`b1.py:3`](encoder_independent_drifting/b1.py#L3)), whose own docstring records
that the finite V-statistic *"is not itself measure determining."* It has never
been tested against a feature-space pressure, and TFD reports its feature-space
coverage term is essential. We keep the clean single factor, but conditions 4
and 5 are **monitored abort criteria during training** (§6.4), not end-of-run
report items, and the anchor-margin arm is **predeclared as the immediate
follow-up**.

Counterargument, recorded: TFD's drifting loss is the *primary* objective while
ASFD's is auxiliary at ~2.5% cadence-averaged influence. But B2's dosing was
comparably light — λ = 1.9×10⁻⁴, one event in ten — and the rank loss was 38–40%
anyway. Light dosing is not the protection it appears to be.

### Stage E — confirmation

Freeze one configuration. Run **two** paired continuation units from the same
foundation checkpoint with new stochastic continuation streams. Access the
sealed CIFAR-10 automobile test split only after both checkpoints are sealed.

Two arms — `EMF-ASFD` and `EMF-raw`. `EMF-control` is established in Stage D and
is not the primary comparison.

**Decision rule — two primary endpoints, both required in both units:**

| primary endpoint | rule |
|---|---|
| KID against `EMF-raw` on the sealed split | lower in **2 of 2** units |
| recall non-inferiority against `EMF-raw` | within a prospectively declared margin in **2 of 2** units |

Everything else is **report-only** and cannot promote or demote: uncurated
fixed-seed one-call grids, class recognizability, FID, precision/density/
coverage, raw and feature-space effective rank, raw and semantic held-out drift
energies, spectrum and multiscale variance, duplicate and nearest-training-image
audits, gradient/cap diagnostics, wall time, memory, training NFEs, and the
inference forward count (which must equal exactly one).

Nine simultaneous conditions across two units is a rule a genuinely better
method can easily fail; two is a real test.

**Two limits that are stated in advance, not discovered:**

- **The two units share one foundation.** CAP-EMF-1 trains a single capability
  unit, so Stage E measures **continuation variance, not foundation variance**.
  No Stage E result can claim replication across foundations.
- **The sealed split is 1 000 images.** B0/B1/B2 used 2 048-sample references
  and three units and were still described as *"coarse consistency, not
  high-powered inference."* Report KID with paired without-replacement
  subsampling over common indices; report FID as **indicative only** — this
  repository has measured its small-sample bias, with a floor near 70 at n=512.

No metric may select a post-hoc checkpoint on the test allocation.

---

## 9. Implementation

### 9.1 Package layout

A new package; the consumed B2 artifacts are not modified.

```text
numerics/encoder_independent_drifting/stage_asfd/
    config.py         frozen dataclasses; every knob that crosses the artifact boundary
    features.py       trunk hooks, token-grid extraction, 66-vector descriptors
    calibration.py    per-channel + scalar normalization, ESS bisection, coefficient calibration
    feature_bank.py   4-view positive/probe banks, hashing, paging
    field.py          normalized Laplace field, energies, separate squares, weight health
    gradients.py      summed gradient, per-component caps, abort criteria, diagnostics
    qualification.py  G1-G9, the t_f grid sweep, Haar band probes, CKA
    preflight.py      Stage C
    development.py    Stage D
    confirmation.py   Stage E
    evaluation.py     metrics, floors, bootstrap/subsampling
    artifacts.py      hashing, provenance, sidecars
    tests/
```

### 9.2 Required tests

Inherited from the original plan:

- forward parity before/after feature hooks;
- deterministic feature-bank hashing;
- no frozen-parameter gradients; **nonzero** generated-input gradients;
- batched-location field agreement with a scalar reference loop;
- exact equality of the multi-radius separate-square implementation to a manual
  sum;
- a counterexample showing aggregate-then-square can cancel;
- ESS calibration for every level × radius;
- sample-role disjointness;
- finite-difference energy gradients;
- AMP unscale before norm measurement;
- checkpoint/resume of every RNG stream;
- same initial state and primary streams across arms;
- inference forward counter equal to one.

Added by this specification:

| test | catches |
|---|---|
| extracted grid reshapes to exactly 16×16 and has nontrivial spatial autocorrelation | the conditioning-token off-by-two slice, invisible to forward parity |
| summed-gradient regression: applied update equals `∇L` to float tolerance | reintroduction of projection |
| per-component cap correctness on **weighted** norms **after** AMP unscale | caps applied to unscaled or unweighted gradients |
| inter-level field cosine and CKA on a synthetic two-level fixture | G8 plumbing |
| per-band Haar sensitivity on a fixture with known band response | G7 plumbing |
| negative-side ESS gate fires on a synthetic degenerate negative cloud | §6.4 |
| real-real floor is deterministic given the allocation, and the sub-floor abort fires | §6.4 |
| ESS calibration fallback ladder engages and is recorded | §5.3 |
| bank hash reproduces under replay at 4 views | §5.7 |
| **no B1 or B2 freeze artifact is loaded anywhere in the ASFD path** | §5.1 |

### 9.3 Artifact discipline

Every artifact records source hashes, environment, CUDA/PyTorch versions,
teacher checkpoint hash, bank hashes, normalization stages and which branch
fired, bandwidths and the ladder rung used, coefficients, streams, allocations,
costs, and acceptance thresholds.

Unit runners refuse to overwrite checkpoints or result artifacts. A crash never
licenses changing a coefficient, bandwidth, threshold, source, arm, or
checkpoint; any such change creates a new development design.

### 9.4 Precision and memory

BF16/FP16 frozen weights; FP32 kernel softmaxes and energy accumulation. If
memory binds, gradient-checkpoint **only** the frozen generated-feature forward.

Do **not**: reduce the raw coefficient, detach generated features, pool to a
single vector, or shrink batches — without rerunning the statistical health gate.

---

## 10. Cost

Correction events cost ≈ 2× an ordinary update — one differentiable frozen-trunk
forward/backward on the generated batch, plus kernels. At one event in ten, a
corrected arm costs **≈ 1.1× per update**.

**Kernel work is not the cost.** At `n_z=64`, `n_p=256`, `n_q=64`, `C=512`:
per level per location ≈ 10.5M MACs, × 66 locations × 4 levels ≈ **5.5 GFLOP**,
with radii sharing the distance matrices. Against a frozen-trunk forward+backward
on 64 images at 256 tokens, depth 12, width 512 (**≈ 2 TFLOP**), that is
**≈ 0.3%**. Four levels and three radii are free relative to the extraction they
ride on. Activation memory is likewise dominated by the trunk's backward graph;
generated-side feature tensors are ≈ 35 MB and distance matrices ≈ 4 MB/level.

In foundation-update-equivalents against CAP-EMF-1's 160 k:

| stage | arms × units × updates | equivalents | fraction of foundation |
|---|---|---:|---:|
| Stage D | 3 × 1 × 20 k | ≈ 66 k | ≈ 41% |
| Stage E, 2 arms | 2 × 2 × 20 k | ≈ 88 k | ≈ 55% |
| **total** | | **≈ 154 k** | **≈ 96%** |

**ASFD roughly doubles the total budget.** Dropping `EMF-control` from Stage E
saves ≈ 28% of foundation cost for no loss of primary inference.

Budget allocation order:

1. foundation capability;
2. one feature qualification + preflight;
3. Stage D;
4. Stage E only for a prospectively selected candidate;
5. final sealed evaluation.

**If the budget will not carry both stages, run Stage D only and report it as a
development-scope result.** That is consistent with how B2.5 and S3R were
reported and is far better than shortening both until neither resolves anything.
If compute runs short, shorten all matched continuation arms equally — **never**
protect the candidate's update count by shortening only its control.

---

## 11. Decision tree

| situation | action |
|---|---|
| **foundation fails** | Stop. The limiting factor is generator capability, not semantic geometry. Spend no time tuning a kernel. |
| **G7 band profile degenerate** | Try §11.1 before canceling. |
| **G8 redundancy failure** | Widen level spacing; re-run. If no spread passes, cancel. |
| **G9 concentration failure after §5.2 stage 1** | The level fails. If all levels fail, cancel. |
| **feature audit fails overall** | Do **not** use external DINO/CLIP to rescue it. Design a separate fixed scattering or random-convolution branch, or improve the foundation. |
| **semantic energy has zero gradient** | Audit detach placement, frozen input Jacobian, feature-noise scale, distance concentration. Do **not** raise the coefficient until the mechanism is mechanically alive. |
| **semantic energy falls but rank/coverage falls** | The §6.4 abort fires during training. Keep the raw/B1 incumbent; the anchor-margin arm is the predeclared follow-up. |
| **raw energy falls below the floor** | Stop the arm. This is estimator exploitation, not distributional improvement, and it is the signature that produced B2's rank collapse. Do not record it as a drift-reduction success. |
| **rank safe but image quality unchanged** | The feature map is non-harmful but uninformative. Inspect neighbour audits and post-cap gradient share. Do **not** call a lower training semantic loss a generative improvement. |
| **ASFD wins** | Replicate on a second class before any general claim. Then test the same frozen protocol on a small multi-class setting. Only after replication investigate a single safeguarded kernel-gradient field or experimentally remove the raw branch. |

### 11.1 Fallback if the trunk is blind

If G7 returns `ρ_HH < 0.25` — the most likely degenerate outcome, since the
trunk allocates capacity under an MSE objective that underweights exactly the
band CAP-EMF-1 is expected to be weakest in — two intermediate options are
preregistered before cancellation:

1. **Decorrelate the feature map from the generator.** Extract from a trunk
   checkpoint that is *not* the fork point: an earlier checkpoint, or an
   independently seeded foundation. The blind spots are correlated with the
   generator's precisely because they are the same weights at the moment the
   branch switches on. Cost: a stored checkpoint, or a second foundation run.
2. **Re-select `t_f`** toward the abstraction end of the grid and re-run G7.
   Cost: one audit pass.

Partial mitigation worth recording: the trunk is trained on **real** noised
images, so its features encode real-image structure, not merely the generator's
output manifold. The self-referential problem is weaker than "the student
teaching itself" — but the band argument survives it, because it concerns
capacity allocation under MSE, not which images were seen.

---

## 12. Claim ledger

### May claim, if Stage E's two primary endpoints hold in both units

> On a preregistered single-class 32×32 pixel task, a one-call EMF generator
> used its own frozen foundation features as a training-only drifting geometry.
> Adding this self-feature field to independent raw identifying and spectral
> anchors improved the matched held-out fidelity/coverage tradeoff without an
> external encoder, VAE, or diffusion teacher.

If only Stage D runs, the claim is the same statement qualified as *"on a single
foundation, at development scope."*

### May not claim

- representation-free generation;
- ImageNet or high-resolution generality;
- that learned features are injective;
- that a finite loss proves distribution equality — the population statement of
  §3.4 is the authority, and it is not what is measured;
- that ASFD is the first use of generative features for drifting;
- superiority to Teacher-Feature Drifting without a matched experiment;
- a formal theorem for the full stochastic, finite-batch training algorithm;
- **replication across foundations** — Stage E's units share one foundation
  checkpoint and measure continuation variance only;
- **that any B1 or B2 confirmation transfers** — every constant is re-derived
  and the raw branch is a new multi-radius configuration;
- **that reduced raw drift energy is itself an improvement**, unless it
  approaches the real-versus-real floor from above.

---

## 13. Build order

1. **Finish B2.5 units 501 and 502** (~11 h local, implemented, paused). It is
   the only measurement of whether the spectral anchor and the raw drift term
   compound, which is ASFD's premise extended to a third term. Unit 500 fails
   two of four preregistered conditions despite `B1B2` leading on recall, KID
   and FID; that ambiguity should be resolved before a third correction term is
   built on top of it.
2. **CAP-EMF-1 foundation.** Record the conditioning decision so extraction can
   be written against it.
3. **Implement `stage_asfd`**, tests first, with G7/G8 and the summed-gradient
   regression as the earliest green targets.
4. **Stage B qualification — G7 and G8 first**, over the `t_f` grid, then banks,
   normalization, bandwidths, coefficients.
5. **Stage C preflight.**
6. **Stage D.**
7. **Stage E only if budget remains**, two arms, two primary endpoints.

Steps 4 and 5 are cheap and target-only. **They are where this design should be
allowed to fail**, and the sequence is arranged so that it can.

---

## 14. Primary sources

- Deng et al., [*Generative Modeling via Drifting*](https://arxiv.org/abs/2602.04770)
- Zhang et al., [*Teacher-Feature Drifting*](https://arxiv.org/html/2605.07327) — verified
- Liu et al., [*Euler Mean Flow*](https://arxiv.org/html/2602.02571)
- Liu et al., [*Learning Deep Kernels for Non-Parametric Two-Sample Tests*](https://proceedings.mlr.press/v119/liu20m.html)
- Lin and Yang, [*Diffusion Model with Perceptual Loss*](https://arxiv.org/html/2401.00110)
- Stracke et al., [*CleanDIFT*](https://openaccess.thecvf.com/content/CVPR2025/html/Stracke_CleanDIFT_Diffusion_Features_without_Noise_CVPR_2025_paper.html)
- Shin et al., [*PixelREPA*](https://arxiv.org/html/2603.14366) — verified
- Esteban-Casadevall et al., [*Kernel-Gradient Drifting Models*](https://arxiv.org/html/2605.10727) — verified
- Li et al., [*MMD GAN*](https://papers.nips.cc/paper_files/paper/2017/hash/dfd7468ac613286cdbb40872c8ef3b06-Abstract.html)
- Bruna and Mallat, [*Invariant Scattering Convolution Networks*](https://arxiv.org/abs/1203.1513)

Local:

- [`AnchoredSelfFeatureDriftingResearchPlan.md`](AnchoredSelfFeatureDriftingResearchPlan.md) — origin, literature survey, novelty assessment
- [`AnchoredSelfFeatureDriftingMechanismAudit.md`](AnchoredSelfFeatureDriftingMechanismAudit.md) — evidence for every change this document makes
- [`EncoderIndependentS3RResultsAndBudgetDecision.md`](EncoderIndependentS3RResultsAndBudgetDecision.md) — CAP-EMF-1 foundation design
- [`EncoderIndependentB0B1B2Results.md`](EncoderIndependentB0B1B2Results.md), [`EncoderIndependentB3Results.md`](EncoderIndependentB3Results.md), [`EncoderIndependentB2Analysis.md`](EncoderIndependentB2Analysis.md)
- [`LaplaceEuclideanConverse.lean`](../DriftingIdentifiability/LaplaceEuclideanConverse.lean), [`FeatureSpaceIdentifiability.lean`](../DriftingIdentifiability/FeatureSpaceIdentifiability.lean)
