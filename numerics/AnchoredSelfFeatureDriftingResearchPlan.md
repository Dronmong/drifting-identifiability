# Anchored self-feature drifting

**Status:** research and implementation plan; no experiment authorized yet

**Date:** 2026-08-03

**Working name:** ASFD (anchored self-feature drifting)
**Primary recommendation:** build this only on top of a successful one-call
EMF foundation; do not replace the raw Laplace field and do not introduce an
external pretrained encoder.

## 1. Executive decision

The raw Euclidean Laplace kernel should not be the only notion of image
similarity used by the eventual model.  It is mathematically valuable but is
an inefficient finite-sample geometry for images: two images can be close in
meaning and far in pixels, while pixelwise closeness can reward blur or local
texture without coherent structure.

The proposed repair is **not** to replace the raw field with DINO, CLIP, an
MAE, or another learned semantic metric.  That would recreate the original
paper's encoder dependence and would surrender source-space identifiability
when the feature map is non-injective.  Instead:

1. train the target one-call EMF generator until it passes a capability gate;
2. clone and freeze its EMA trunk as a target-specific feature map;
3. retain an independently squared raw-pixel Laplace drift energy with a
   strictly positive coefficient;
4. add independently squared feature-space drift energies at frozen
   intermediate layers and target-only calibrated bandwidths; and
5. use the existing spectral anchor and gradient protection to prevent the
   semantic branch from buying apparent fidelity by losing coverage.

The population correction has the form

\[
\mathcal L_{\rm corr}
=\lambda_{\rm spec}\mathcal L_{\rm B1}
+\lambda_{\rm raw}\mathcal E_{\rm raw}
+\lambda_{\rm self}\mathcal E_{\rm self},
\qquad
\lambda_{\rm raw}>0,
\]

where every displayed component is nonnegative and

\[
\mathcal E_{\rm self}
=\frac{1}{|J||R|}\sum_{j\in J}\sum_{r\in R}
 \mathbb E_z\|V^{\phi_j}_{r}(z)\|_2^2.
\]

The fields from different layers and bandwidths are **squared before they are
averaged**.  Squaring an aggregate field would permit cancellation between
two incorrect fields and would weaken the zero-set interpretation.

This is the strongest defensible route because it simultaneously provides:

- image-aware finite-sample neighborhoods;
- no externally pretrained encoder or diffusion teacher;
- no feature network at inference;
- one model call at inference;
- a raw source-space anchor connected to the formal theorem; and
- a clean ablation against both EMF alone and the existing raw correction.

## 2. What problem this actually solves

The original drifting paper states that its feature extractor is important in
high-dimensional image generation and reports that ImageNet training did not
work without one.  Its best setup separately pretrains a latent MAE and then
drifts in the resulting multi-scale feature space.  This gives useful
similarities but makes model quality depend on a separate representation
authority.

There are three distinct notions that must not be conflated:

1. **No external encoder (E1).** No DINO, CLIP, MoCo, SimCLR, MAE, VAE encoder,
   or diffusion teacher supplies training geometry.
2. **No separate learned representation (E2).** No separately optimized
   network decides which distributions are equal.
3. **Correctness independent of learned features (E3).** Even if the learned
   feature map collapses two different images, a separate source-space term
   still prevents that collision from becoming an exact equilibrium.

ASFD targets all three in a precise but limited sense.  Its frozen feature
trunk is learned jointly while building the generator foundation, so the
method does use an internal representation.  It does **not** claim to be
representation-free.  It claims:

> no external or separately pretrained representation is needed, and exact
> population correctness does not rely on the self-feature map being
> injective.

That is stronger and more honest than calling any architecture containing
hidden states "encoder-free."

## 3. Evidence from this repository

### 3.1 Why the raw term stays

[`laplaceZeroDrift_identifies_euclidean`](../DriftingIdentifiability/LaplaceEuclideanConverse.lean)
proves, in the formalized population setting, that the full Euclidean Laplace
field identifies arbitrary probability measures.  The feature-space collision
result in
[`FeatureSpaceIdentifiability.lean`](../DriftingIdentifiability/FeatureSpaceIdentifiability.lean)
shows why a non-injective feature map cannot supply the same source-law claim.

If the population probe law has full support, the raw field is continuous,
and

\[
\mathcal E_{\rm raw}=\int\|V^{\rm raw}_{p,q}(z)\|^2\,d\rho(z)=0,
\]

then the existing energy/continuity bridge promotes almost-everywhere zero to
pointwise zero, after which the Laplace converse gives \(p=q\).  Therefore

\[
\mathcal L_{\rm corr}=0\Longrightarrow
\mathcal E_{\rm raw}=0\Longrightarrow p=q
\]

as long as \(\lambda_{\rm raw}>0\).  No injectivity assumption on the
self-feature map appears in this implication.

This is an exact population statement, not a claim that a finite minibatch
loss or a nonzero trained loss identifies the learned law.

### 3.2 Why raw alone is not enough operationally

Stage B2 showed that the raw normalized-Laplace energy is a real, useful
gradient signal:

- paired held-out raw-energy reductions were about 23--27%;
- all 18 paired audit comparisons improved; and
- coverage did not collapse catastrophically.

It also exposed the main practical weakness:

- raw-pixel effective rank fell by about 38--40%;
- FID/KID were mixed; and
- rare rows had very small effective neighborhoods even though median ESS was
  well calibrated.

The raw field therefore deserves to remain as a **correctness anchor**, but it
has not earned the role of sole image geometry.

### 3.3 What the earlier encoder ladder does and does not say

Phases 17--18 found that a pretrained ResNet was much worse than raw pixels
when feature distances were used only to weight **pixel-space displacement**.
An untrained ResNet was near raw.  This is a warning against an invariant
semantic kernel controlling unconstrained pixel directions.

It is not a test of feature-space drifting.  ASFD computes both distances and
drift vectors inside the frozen feature space and backpropagates through that
space.  The prior negative therefore remains a useful safety warning but does
not directly adjudicate this proposal.

### 3.4 Why the generator must work first

The S3R mechanism screen selected direct-\(x\) Euler Mean Flow (EMF) as the
strongest one-call foundation, but it did not yet produce a promoted image
generator.  A failed foundation cannot be expected to contain useful visual
features.  ASFD is consequently conditional on a successful CAP-EMF
foundation checkpoint.  If the foundation fails the recognizable-image and
diversity gate, this plan stops before feature extraction.

## 4. What the literature establishes

### 4.1 Deep kernels improve finite-sample discrimination, but need a safeguard

[Liu et al., *Learning Deep Kernels for Non-Parametric Two-Sample
Tests*](https://proceedings.mlr.press/v119/liu20m.html) show that learned deep
kernels can adapt to spatially heterogeneous differences in complex,
high-dimensional distributions and improve two-sample test power.  They also
state the exact danger relevant here: a characteristic kernel applied only to
\(\phi(x)\) is characteristic on the source only when \(\phi\) is injective.
Their safeguarded construction

\[
k_\omega(x,y)=\big[(1-\epsilon)\kappa(\phi_\omega(x),\phi_\omega(y))
+\epsilon\big]q(x,y)
\]

retains a characteristic base kernel \(q\).

ASFD borrows the safeguard principle, not that exact product kernel.  The
project's converse is for a normalized Laplace mean-shift field, and it would
be incorrect to assume without proof that the same theorem transfers to the
deep product kernel.  Keeping a separate nonnegative raw energy is the safer
analogue: its zero cannot be canceled or hidden by the learned feature branch.

[MMD-GAN](https://papers.nips.cc/paper_files/paper/2017/hash/dfd7468ac613286cdbb40872c8ef3b06-Abstract.html)
and later MMD work support the practical premise that a learned kernel can be
far more useful than a stationary raw kernel on images.  They also show why a
live adversarial kernel is not the first experiment: jointly learning the
metric introduces a second game, biased generator gradients after fitting the
critic on finite samples, and new stability questions.  ASFD freezes its
metric before the correction fork.

### 4.2 A generative model can supply its own perceptual features

[Lin and Yang, *Diffusion Model with Perceptual
Loss*](https://arxiv.org/html/2401.00110), train a normal diffusion model,
freeze a copy, and use its hidden activations as a self-perceptual objective.
The key implementation results are directly relevant:

- the frozen copy is preferable to a moving EMA teacher;
- the midblock was better than indiscriminately summing every layer; and
- applying the frozen network at sampled noisy states was better than a
  mismatched single state.

Their objective is paired perceptual regression in a multi-step diffusion
model, not distributional drifting in a one-call EMF model.  It is precedent
for the self-teacher mechanism, not a prior instance of ASFD.

[Generic Perceptual Loss](https://arxiv.org/abs/2103.10571) further shows that
even a properly initialized random convolutional hierarchy can encode useful
structured-output dependencies.  This helps explain the repository's random
ResNet result and supplies a fixed-architecture fallback, but it is weaker
evidence for semantic neighborhoods than a successful target-trained
foundation.

### 4.3 Clean features are not automatically the best features

[CleanDIFT](https://openaccess.thecvf.com/content/CVPR2025/html/Stracke_CleanDIFT_Diffusion_Features_without_Noise_CVPR_2025_paper.html)
shows that ordinary diffusion features are often poor on clean inputs because
the network was trained on noisy states, and that merely averaging arbitrary
noise draws does not repair a badly matched extraction regime.

[Deep MMD Gradient Flow](https://arxiv.org/abs/2405.06780) likewise finds value
in noise-adaptive discrepancies.  These results make a clean \(t=0\) feature
map an unjustified default for a time-conditioned EMF trunk.

ASFD instead evaluates a real or generated image through the foundation's
native corruption path:

\[
x_{t_f}=(1-t_f)x+t_f\xi,
\qquad
\phi_j(x;\xi)=h^{(j)}_{\theta_*}(x_{t_f},t_f,0),
\]

with frozen \(t_f=0.10\), independent noise streams, and interval zero.  This
is an in-distribution boundary/denoising condition for the EMF trunk, not an
arbitrary additive perturbation.

### 4.4 Teacher-feature drifting validates the mechanism and narrows novelty

The closest work is the May 2026 paper
[Teacher-Feature Drifting](https://arxiv.org/html/2605.07327).  It removes the
original drifting paper's separate encoder by using intermediate features of
a pretrained diffusion teacher.  It reports:

- useful clean teacher features and better results with moderate noise;
- better results from selected encoder, bottleneck, and decoder levels than
  from a poorly chosen layer subset;
- a multi-radius feature-space drift; and
- an anchor-margin term that improves within-class coverage.

This is strong evidence that feature-space drift over generative hidden states
can work.  It also means that "replace the encoder with generative-model
features" is no longer by itself novel.

The remaining distinction is substantive:

| Teacher-Feature Drifting | ASFD proposal |
|---|---|
| starts from a pretrained multi-step diffusion teacher | trains a one-call EMF foundation from target data |
| distills the teacher and inherits its geometry | uses a frozen copy of the generator's own foundation |
| no extra encoder, but a powerful teacher is required | no external teacher or separately pretrained representation |
| feature geometry carries the practical objective | an independent raw field carries the exact source-space anchor |
| coverage uses a teacher-feature anchor margin | initial ASFD uses the repository's raw spectral B1 protector |

The honest novelty question is therefore whether **self-founded**, rather
than teacher-distilled, feature drifting can improve a one-call generator
without making correctness depend on the learned feature map.

### 4.5 Why the semantic branch cannot dominate

[PixelREPA](https://arxiv.org/html/2603.14366) shows that forcing a
pixel-space generator toward a compressed semantic target can worsen FID and
collapse diversity.  Many pixel-distinct images share nearly identical
semantic features, making direct feature regression a shortcut.

ASFD addresses that warning in four ways:

1. the semantic term is a distributional drift energy, not direct per-image
   regression to a compressed target;
2. local spatial tokens and global statistics are retained instead of using
   only one pooled vector;
3. raw Laplace and raw spectral terms remain independently active; and
4. semantic gradients are capped and prevented from opposing the primary EMF
   gradient to first order.

These are safeguards, not a proof that feature hacking cannot occur.  Relative
rank, coverage, and memorization gates remain mandatory.

### 4.6 Longer-term kernel-gradient option

[Kernel-Gradient Drifting](https://arxiv.org/html/2605.10727) replaces the
Euclidean displacement by \(\nabla_x k(x,y)\), obtains a score-ratio field for
general kernels, and proves identifiability for characteristic smoothing
kernels.  This is a promising later route for a single semantic/raw
safeguarded kernel.

It is not the first implementation here.  A self-feature product kernel is
generally not translation invariant, its normalizing measure can depend on
\(x\), and the project's existing Laplace converse cannot simply be reused.
The dual-energy ASFD experiment requires fewer unproved mathematical and
software changes.  Kernel-gradient ASFD should be attempted only after the
frozen self-feature geometry shows a measurable benefit.

### 4.7 Fixed analytic fallback

[Wavelet scattering](https://arxiv.org/abs/1203.1513) supplies a fixed
translation-stable, deformation-stable hierarchy that retains high-frequency
information.  It is a useful no-learning fallback if the self-feature audit
fails.  It should be a separately frozen experiment, not silently substituted
during ASFD confirmation.

## 5. Exact proposed mechanism

### 5.1 Foundation and frozen teacher

Let \(f_\theta\) be the one-call EMF generator and let \(\theta_*\) be the EMA
weights at the successful foundation checkpoint.  Create a separate module
\(h_{\theta_*}\) with:

- all parameters `requires_grad = False`;
- evaluation mode;
- a recorded source-checkpoint hash;
- frozen feature-normalization statistics; and
- no updates during either continuation.

Freezing parameters must **not** detach generated inputs.  The generated
branch needs \(\partial h_{\theta_*}(x)/\partial x\) so the semantic energy can
reach \(\theta\).  Implementing the frozen forward under `torch.no_grad()`
would silently kill the proposed mechanism.

The primary one-call generator remains the only exported inference object.
The frozen feature module, feature banks, kernels, B1, and raw probes are all
training-only.

### 5.2 Feature levels

Expose hidden tokens without changing the ordinary forward calculation.  For
a patch-2, depth-12 U-shaped transformer, predeclare two levels:

1. `bottleneck`: output of the last encoder block;
2. `early_decoder`: output of the first decoder block after its long-skip
   fusion.

For each level, reshape the 16-by-16 tokens and construct:

- 8-by-8 local vectors by non-overlapping 2-by-2 average pooling;
- one global channel mean vector; and
- one global channel standard-deviation vector.

This yields 66 vectors per level while preserving local structure.  Do not
add every layer.  Both the self-perceptual and TFD ablations show that feature
selection matters and that more layers are not monotonically better.

### 5.3 Native feature noising

Use \(t_f=0.10\).  For each image and each feature draw:

\[
x_{t_f}=0.9x+0.1\xi,
\quad \xi\sim\mathcal N(0,I),
\]

then evaluate the frozen trunk at absolute time `0.10` and interval `0`.
Positive, probe, and generated-negative roles receive independent random
streams.  Do not pair a positive and negative through the same noise tensor.

The first implementation has one feature-noise level.  A time grid is deferred
because changing feature time, layer set, and kernel together would make a
small experiment uninterpretable.

### 5.4 Fixed target normalization

For feature level \(j\) of channel width \(C_j\), estimate on a target-only
calibration allocation

\[
S_j=\frac{1}{\sqrt{C_j}}
\operatorname{mean}_{a\ne b,\ell}
\|h_j(x_a)_\ell-h_j(x_b)_\ell\|_2.
\]

Use \(\widetilde h_j=h_j/S_j\) and freeze \(S_j\).  All locations within one
feature level share a scale.  Reject zero, nonfinite, or poorly resolved
scales.  Fixed target scales are preferable to recomputing a batch scale from
the evolving generator because they prevent the metric itself from moving in
response to model failure.

### 5.5 Semantic Laplace fields

For every feature level, spatial/statistical location, and bandwidth, use the
same sample-split normalized Laplace field as B2:

\[
\widehat V^{j,r}(z)=
\frac{\sum_a e^{-\|z-y_a^+\|/\tau_{j,r}}y_a^+}
     {\sum_a e^{-\|z-y_a^+\|/\tau_{j,r}}}
-
\frac{\sum_b e^{-\|z-y_b^-\|/\tau_{j,r}}y_b^-}
     {\sum_b e^{-\|z-y_b^-\|/\tau_{j,r}}}.
\]

Vectorize locations as a leading batch dimension.  The energy is

\[
\mathcal E_{\rm self}
=\operatorname{mean}_{j,r,\ell,z}
\frac{1}{C_j}\|\widehat V^{j,r}_\ell(z)\|_2^2.
\]

Use stable row softmaxes.  Positive and probe features are detached; generated
negative features remain differentiable through the frozen trunk into the
online generator.

### 5.6 Bandwidths

Calibrate target-only bandwidths separately for each feature level using
self-excluded effective sample size.  Predeclare three neighborhood regimes:

\[
R=\{0.35,0.60,0.85\}
\]

as target median ESS fractions.  This translates the original paper's and
TFD's multi-temperature lesson into the scale-free statistic already
validated by B2.

Do not add the three fields and square the sum.  Average their three squared
energies.  Log, per level and radius:

- ESS median, 5th percentile, and minimum;
- maximum-weight median and 95th percentile;
- distance median and coefficient of variation;
- field RMS and energy; and
- row-sum numerical error.

Reject a calibration if the 5th-percentile ESS is below `0.10` or the
95th-percentile maximum weight exceeds `0.50`.  These values are prospective
health limits motivated by B2's observed one-to-five-neighbor tail, not
post-hoc quality tuning.

### 5.7 Target feature banks

After freezing \(\theta_*\), precompute two independently noised feature views
per training image for the positive bank and two disjoint views for the probe
bank.  Record image indices, augmentation bits, feature-noise seeds, feature
configuration, teacher hash, dtype, normalization scales, and bank hashes.

This avoids two frozen-trunk forwards on every correction event.  The
generated branch is always recomputed with fresh feature noise and remains
differentiable.  Evaluation uses a separately seeded fresh feature bank so a
model cannot pass merely by matching the cached views.

## 6. Feature qualification gate

The self-feature branch is not activated merely because a checkpoint exists.
Run the following training-only audit on the frozen feature map.

### 6.1 Mechanical checks

1. Adding `extract_feature_pyramid` changes ordinary model outputs by at most
   the declared mixed-precision tolerance.
2. Frozen parameters receive no gradients.
3. Generated images receive finite, nonzero input gradients through every
   selected feature level.
4. Replaying hashes and noise streams reproduces cached features.
5. CPU float64 and GPU calculations agree on a tiny deterministic case.

### 6.2 Geometry checks

On target-training images only:

1. benign-pair distance is below random-pair distance with AUC at least
   `0.80`.  Compute this once on the invariant global mean/std descriptor and
   once on local tokens after undoing the known horizontal flip and spatial
   translation.  Do **not** compare unregistered token locations, which would
   mistake a coordinate permutation for a semantic failure;
2. patch-shuffled or phase-scrambled images are farther than benign variants
   in at least `80%` of paired cases;
3. global feature effective rank is at least `16`, and no single principal
   component explains more than `50%` of variance;
4. pairwise-distance coefficient of variation is at least `0.05`;
5. the response to small high-frequency noise is not more than four times the
   response of a normalized raw-pixel control; and
6. the semantic field is not numerically identical to the raw field
   (`|cosine| < 0.995` at the median audit event).

The thresholds reject collapse, a flat kernel, and the hypersensitivity seen
in the earlier pretrained-ResNet harness.  They do not prove human semantic
alignment.  Record uncurated training-only nearest-neighbor grids as a sanity
check, not as a threshold-selection device.

If this gate fails, do not tune layers on held-out image quality.  The ASFD
arm is canceled.  A separately preregistered scattering or random-convolution
experiment may then be designed.

## 7. Gradient integration

### 7.1 Components

On each correction event compute:

- \(g_0=\nabla\mathcal L_{\rm EMF}\);
- \(g_1=\nabla\mathcal L_{\rm B1}\);
- \(g_r=\nabla\mathcal E_{\rm raw}\); and
- \(g_s=\nabla\mathcal E_{\rm self}\).

Use training-only calibration events to freeze numerical coefficients.  The
candidate event-level unprojected ratios are initially

\[
\|\lambda_1g_1\|/\|g_0\|=0.15,\quad
\|\lambda_rg_r\|/\|g_0\|=0.10,\quad
\|\lambda_sg_s\|/\|g_0\|=0.10.
\]

These are not simply summed without protection.  Project each auxiliary
component so it does not oppose \(g_0\) to first order, then cap the norm of the
combined auxiliary update at `0.25 * ||g0||`.  Apply correction once every ten
updates.  This bounds its nominal cadence-averaged influence near 2.5%.

The raw-only incumbent uses the same B1/raw coefficients and the same final
combined cap.  ASFD adds the semantic component but cannot exceed the same
maximum auxiliary norm.  Log post-projection and post-cap component shares so
an apparent win cannot conceal that the raw anchor was numerically erased.

### 7.2 Mandatory diagnostics

At every logged correction event record:

- each loss and pre-projection gradient norm;
- all pairwise gradient cosines;
- projection magnitude;
- cap activation and post-cap norm;
- final component contribution estimates;
- ordinary gradient-clipping status;
- feature and raw kernel health; and
- frozen-teacher input-gradient norm.

Abort the ASFD arm on nonfinite gradients, a broken feature-bank hash, or a
zero generated-input Jacobian.  Do not abort on one negative cosine; the
projection is designed to handle it.

## 8. Experimental sequence

### Stage A — one-call foundation

Implement and run CAP-EMF-1 without drift.  The foundation must pass all of:

- visibly recognizable uncurated automobiles;
- nontrivial classifier recognizability on training-only diagnostics;
- no duplicate/memorization veto;
- adequate raw variance and high-frequency retention;
- stable one-call forward counter; and
- mature EMA by the declared checkpoint.

If Stage A fails, ASFD stops.  Hidden states of a failed generator are not a
credible semantic metric.

### Stage B — frozen-feature qualification

Freeze the successful EMA trunk, expose the two feature levels, create banks,
calibrate scales and ESS bandwidths, and run Section 6.  No official test
image is accessed.

### Stage C — implementation preflight

Before a long continuation:

1. test feature extraction parity;
2. test field values on equal, shifted, and collapsed toy feature laws;
3. finite-difference the semantic energy with respect to generated features;
4. verify gradients reach online generator parameters only through generated
   negatives and the frozen input Jacobian;
5. verify separate-squares behavior with two canceling synthetic fields;
6. calibrate all component weights outcome-blind;
7. run 500 ordinary updates and 50 correction events; and
8. project wall time and peak memory before continuing.

### Stage D — paired development fork

Clone exactly the same successful foundation into three arms:

| Arm | Continuation |
|---|---|
| `EMF-control` | EMF only |
| `EMF-raw` | EMF + B1 + raw Laplace energy |
| `EMF-ASFD` | EMF + B1 + raw Laplace energy + frozen self-feature energy |

Use the same flow batches, latents, EMF time samples, positive indices, update
count, optimizer state, EMA state, and evaluation cohorts.  Auxiliary streams
are role-separated and deterministic.  Run a short development continuation
first; no test images are used for arm selection.

The semantic arm advances only if, relative to `EMF-raw`, it:

- reduces fresh-bank semantic energy;
- does not worsen fresh-bank raw energy by more than `5%`;
- retains at least `90%` of the incumbent effective rank;
- retains precision and recall within prospective uncertainty margins;
- shows no increase in duplicates or nearest-training-image concentration;
- has a nonzero semantic post-cap gradient share; and
- has a projected training cost compatible with the remaining budget.

### Stage E — confirmation

Freeze one configuration.  Run two paired continuation units from the same
foundation checkpoint but with new stochastic continuation streams.  Access
the untouched CIFAR-10 automobile test split only after both checkpoints are
sealed.

Primary comparisons are `EMF-ASFD` versus both `EMF-control` and `EMF-raw`.
Report:

- fixed-seed uncurated one-call grids;
- class recognizability;
- KID and explicitly finite-sample FID;
- improved precision/recall and density/coverage;
- raw-pixel and feature-space effective rank;
- raw and semantic held-out drift energies;
- spectrum and multiscale variance;
- duplicate and nearest-training-image audits;
- gradient/cap diagnostics;
- wall time, memory, training NFEs; and
- inference forward count exactly equal to one.

No metric is allowed to select a post-hoc checkpoint on the test allocation.

## 9. Cost controls

The frozen model roughly duplicates parameter storage, but target features are
cached and correction is sparse.  The expensive semantic work per correction
event is therefore primarily:

1. one online one-call generation for the independent negative batch; and
2. one differentiable frozen-trunk feature extraction on that generated
   batch.

Kernel work over 66 locations and three bandwidths is small compared with the
transformer forward.  A faithful preflight must measure, not assume, this.

Use BF16/FP16 frozen weights with FP32 kernel softmaxes and energy accumulation.
If memory is high, checkpoint only the frozen generated-feature forward.  Do
not reduce the raw coefficient, detach generated features, silently pool to a
single vector, or shrink batches before rerunning the statistical health gate.

The long-run budget is allocated in this order:

1. foundation capability;
2. one feature qualification/preflight;
3. short three-arm development fork;
4. two-arm confirmation only for a prospectively selected candidate; and
5. final sealed evaluation.

If compute runs short, shorten all matched continuation arms equally.  Never
protect the candidate's update count by shortening only its control.

## 10. Implementation map

Create a new package rather than modifying the consumed B2 artifacts:

```text
numerics/encoder_independent_drifting/stage_asfd/
    config.py
    features.py
    feature_bank.py
    field.py
    gradients.py
    preflight.py
    development.py
    confirmation.py
    evaluation.py
    artifacts.py
    tests/
```

Required tests include:

- forward parity before/after feature hooks;
- deterministic feature-bank hashing;
- no frozen-parameter gradients and nonzero generated-input gradients;
- batched-location field agreement with a scalar reference loop;
- exact equality of multi-radius separate-square implementation to a manual
  sum;
- a counterexample showing aggregate-then-square can cancel;
- ESS calibration for every feature level/radius;
- sample-role disjointness;
- finite-difference energy gradients;
- AMP unscale before norm measurement/projection;
- checkpoint/resume of every RNG stream;
- same initial state and primary streams across arms; and
- inference forward counter equal to one.

Every artifact records source hashes, environment, CUDA/PyTorch versions,
teacher checkpoint hash, bank hashes, normalization, bandwidths, coefficients,
streams, allocations, costs, and acceptance thresholds.

## 11. Decision tree

### If the foundation fails

Stop.  Spend no time tuning a self-feature kernel.  The limiting factor is
generator capability, not semantic geometry.

### If the feature audit fails

Do not use external DINO/CLIP to rescue the declared experiment.  Design a
separate fixed scattering or random-convolution branch, or improve the
foundation.

### If self features are healthy but semantic energy has zero gradient

Audit detach placement, the frozen input Jacobian, feature-noise scale, and
distance concentration.  Do not increase the coefficient until the mechanism
is mechanically alive.

### If semantic energy falls but rank/coverage falls

The PixelREPA warning has materialized.  Keep the raw/B1 incumbent.  A later
experiment may test TFD-style anchor margin or a masked adapter, one factor at
a time.

### If rank is safe but image quality is unchanged

The feature map is non-harmful but uninformative.  Inspect feature-neighbor
audits and post-cap gradient share.  Do not call a lower training semantic
loss a generative improvement.

### If ASFD wins

Replicate on a second class before any general claim.  Then test the same
frozen protocol on a small multi-class setting.  Only after replication should
the project investigate a single safeguarded kernel-gradient field or remove
the raw branch experimentally.

## 12. Claim ledger

### A successful first experiment may claim

> On a preregistered single-class pixel task, a one-call EMF generator used
> its own frozen foundation features as a training-only drifting geometry.
> Adding this self-feature field to independent raw identifying and spectral
> anchors improved the matched held-out fidelity/coverage tradeoff without an
> external encoder, VAE, or diffusion teacher.

### It may not claim

- representation-free generation;
- ImageNet or high-resolution generality;
- that learned features are injective;
- that a finite loss proves distribution equality;
- that ASFD is the first use of generative features for drifting;
- superiority to Teacher-Feature Drifting without a matched experiment; or
- a formal theorem for the full stochastic, finite-batch training algorithm.

## 13. Novelty assessment

The ingredients individually exist:

- learned deep kernels;
- self-perceptual frozen generative features;
- teacher-feature drifting;
- raw characteristic safeguards; and
- one-call MeanFlow-style generators.

The apparently underexplored intersection is:

1. no external pretrained representation **and no pretrained diffusion
   teacher**;
2. a one-call generator supplies its own frozen feature geometry;
3. feature-space drift is used as an auxiliary distributional correction,
   not a paired perceptual target;
4. a separately squared, formally identifying source-space field prevents
   semantic collisions from defining equilibrium; and
5. coverage protection and gradient caps are audited explicitly.

This is a plausible research contribution, not a guaranteed novelty claim.
A final paper search and direct comparison with TFD and contemporaneous
kernel-gradient work are required before publication language is frozen.

## 14. Recommended immediate action

Do **not** implement the semantic correction first.  The immediate order is:

1. finish the CAP-EMF-1 foundation run card and capability implementation;
2. add feature-hook parity support while leaving ordinary forward unchanged;
3. run the foundation;
4. only if it succeeds, implement Sections 5--7 and the feature gate;
5. run the short three-arm development fork; and
6. authorize confirmation only after the prospective gate passes.

This order prevents another long experiment from confounding a weak generator
with a weak kernel.  It also makes the semantic proposal cheap to abandon if
the prerequisite representation never materializes.

## 15. Primary sources

- Deng et al., [*Generative Modeling via Drifting*](https://arxiv.org/abs/2602.04770).
- Zhang et al., [*Teacher-Feature Drifting*](https://arxiv.org/html/2605.07327).
- Liu et al., [*Learning Deep Kernels for Non-Parametric Two-Sample Tests*](https://proceedings.mlr.press/v119/liu20m.html).
- Li et al., [*MMD GAN*](https://papers.nips.cc/paper_files/paper/2017/hash/dfd7468ac613286cdbb40872c8ef3b06-Abstract.html).
- Binkowski et al., [*Demystifying MMD GANs*](https://arxiv.org/abs/1801.01401).
- Lin and Yang, [*Diffusion Model with Perceptual Loss*](https://arxiv.org/html/2401.00110).
- Stracke et al., [*CleanDIFT*](https://openaccess.thecvf.com/content/CVPR2025/html/Stracke_CleanDIFT_Diffusion_Features_without_Noise_CVPR_2025_paper.html).
- Shin et al., [*Representation Alignment for Just Image Transformers Is Not Easier Than You Think*](https://arxiv.org/html/2603.14366).
- Galashov et al., [*Deep MMD Gradient Flow without Adversarial Training*](https://arxiv.org/abs/2405.06780).
- Esteban-Casadevall et al., [*Kernel-Gradient Drifting Models*](https://arxiv.org/html/2605.10727).
- Liu et al., [*Generic Perceptual Loss for Modeling Structured Output Dependencies*](https://arxiv.org/abs/2103.10571).
- Bruna and Mallat, [*Invariant Scattering Convolution Networks*](https://arxiv.org/abs/1203.1513).
