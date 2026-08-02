# Single-Class Transformer Capability Experiment

**Status:** evidence-vetted implementation protocol; implementation and the
long run have not started.  The initial proposal is recorded first.  The
detailed protocol in Section 2 onward is authoritative wherever it differs,
with Section 18's CAP-1R revision authoritative for timestep sampling and its
associated diagnostics.

## 1. Initial proposal (recorded before the detailed research pass)

### Objective

Use a deliberately easier but nontrivial image-generation problem to answer:

> Can the strongest generator that fits the local one-day compute budget learn
> recognizable, diverse images, and can the formalization-motivated B1/B2
> corrections improve that working generator?

The initial task is one CIFAR-10 class, provisionally `automobile`: 5,000 train
images, 1,000 held-out test images, native 32x32 resolution, and horizontal
flips only.  This is a capability gate, not evidence of ten-class or ImageNet
generality.

### Initial architecture proposal

- time-conditioned rectified-flow transformer;
- approximately 25--35M parameters;
- 12 transformer blocks, width 384, six attention heads;
- patch size 2 if a throughput-only preflight fits the budget, otherwise 4;
- adaLN-zero time conditioning;
- RMSNorm, QK normalization, SwiGLU, rotary position embeddings;
- a small convolutional output-refinement head;
- activation checkpointing, scaled-dot-product attention, mixed precision;
- EMA readouts at 0.999 and 0.9999.

### Initial staged design

1. Train a pure flow-matching foundation for roughly 14--16 hours.
2. If it is recognizable, fork the identical checkpoint:
   - control: continue pure flow matching;
   - candidate: continue with
     `L_flow + lambda_1 L_B1 + lambda_2 L_balanced-B2`.
3. Allocate roughly three hours to each branch and one to two hours to final
   sampling and evaluation.
4. If the foundation is not recognizable, do not activate experimental
   corrections; spend the remaining training budget on the foundation.

The candidate retains B1's global spectral protection and replaces B2's
single row-normalized raw-pixel field with a conservative multiscale,
multibandwidth, two-sided-balanced correction.  B1 and B2 coefficients must be
recalibrated at the fork and must not be copied from the earlier U-Net runs.

### Initial evidence requirements

Report fixed-seed uncurated grids, classifier recognizability, KID, indicative
class-conditional FID, precision/recall, effective rank, duplicates,
nearest-training-image checks, runtime, memory, and NFE 1/8/32 samples.  The
1,000-image held-out class subset is too small to treat FID as an exact primary
endpoint; KID, uncertainty, visual evidence, and memorization controls are
required.

## 2. Detailed evidence-vetted protocol

This is the authoritative protocol.  It supersedes the provisional choices in
Section 1 where the two differ.

### 2.1 Executive decision

Run one experiment, provisionally named **CAP-1**, with a hard limit of one day
of training on the local RTX 4050 Laptop GPU:

1. train a strong encoder-free pixel-space flow model for 100,000 updates;
2. require that this shared foundation already produces recognizable and
   non-collapsed automobiles;
3. fork the exact checkpoint, optimizer, EMA, data order, and flow-noise
   streams into two 25,000-update continuations;
4. continue one arm with flow matching alone;
5. continue the other with the same flow updates plus a conservative,
   gradient-protected B1/B2 geometry correction;
6. evaluate both arms once on the untouched CIFAR-10 automobile test split,
   using the same latent samples and identical reporting code.

The generator is a compute-scaled, pixel-space **U-ViT-S/2 flow model**:
26.1M parameters, 384 hidden channels, six heads, 256 image tokens, long
U-shaped transformer skips, and a convolutional refinement head.  Training
uses a linear stochastic interpolant and velocity prediction.  Noise/data
pairs are coupled by exact minibatch assignment using a fixed multiscale
low-frequency cost.

This is deliberately not a literal reproduction of either U-ViT, SiT, or the
paper's one-step drifting generator.  Every constituent has evidence, but the
combination is a new engineering hypothesis.  The within-run control is what
will tell us whether the B1/B2 addition helps that working hypothesis.

### 2.2 Precise question and claim boundary

CAP-1 asks two nested questions:

1. **Capability:** can an encoder-free transformer trained from scratch on one
   32x32 class produce recognizable, diverse images within one local GPU day?
2. **Mechanism:** once that generator works, does a formalization-motivated
   correction improve held-out distributional quality or drift discrepancy
   without compressing its geometry?

A positive run may support a statement such as:

> On a prespecified one-class CIFAR-10 pilot, a 26M-parameter pixel-space flow
> transformer produced recognizable images without a pretrained training
> encoder; the protected geometry correction [did/did not] improve its paired
> held-out tradeoff.

It may not support any of the following:

- parity with the paper's ImageNet model;
- a ten-class, high-resolution, or general image-generation claim;
- a claim that an external feature encoder is never useful;
- a claim that a finite minibatch drift loss proves population
  identifiability;
- a published FID comparison based on only 1,000 held-out images;
- a training-seed significance claim, because this pilot has one shared seed.

Evaluation networks are allowed only after or outside training.  They do not
define the generator loss, the transport pairing, a bandwidth, or an update.
This is encoder-independent training in the relevant sense, not
encoder-independent measurement.

## 3. Why this is the strongest defensible one-day design

### 3.1 What the repository already establishes

The local evidence is unusually informative and rules out several tempting
but bad choices:

| Evidence | Consequence for CAP-1 |
|---|---|
| B0's prescribed flow reached recall 0.15--0.20 while one-step drifting proxies remained near zero | Build a reliable bridge before adding drift |
| B1 improved the best historical matched KID/FID and retained effective rank | Retain B1 as the global geometry protector |
| B2 reduced held-out raw drift by 23--27%, but reduced effective rank by 38--40% | Use B2 only after the model works, at a smaller protected gradient budget |
| B3's 26x capacity increase did not repair its one-step drifting proxy | Capacity alone cannot rescue the wrong objective |
| B2 correction events reached roughly 5.49 GB and made training 1.7x slower | Share one generated trajectory between B1 and B2 and lower correction NFE |
| Phase 22 showed that FID/KID can disagree with visual fidelity and coverage | Require uncurated images, precision/recall, spectrum, and memorization checks |
| F1 showed direct iteration of the raw drift map collapses even real data | Penalize a nonnegative discrepancy through parameters; never use raw drift as the sampler |

The detailed local records are
[`EncoderIndependentB0B1B2Results.md`](EncoderIndependentB0B1B2Results.md),
[`EncoderIndependentB2Analysis.md`](EncoderIndependentB2Analysis.md), and
[`EncoderIndependentB3Results.md`](EncoderIndependentB3Results.md).

### 3.2 Why a U-shaped transformer

[U-ViT](https://arxiv.org/abs/2209.12152) directly studied pixel-space CIFAR-10
and found that long shallow-to-deep transformer skips were crucial, while a
3x3 convolutional output head further improved visual quality.  Its official
[CIFAR configuration](https://github.com/baofff/U-ViT/blob/main/configs/cifar10_uvit_small.py)
uses patch size 2, width 512, six input blocks, a middle block, six output
blocks, and 500,000 updates.  That exact 45M-scale schedule is too expensive
locally; preserving its topology and reducing width to 384 is a cleaner
compute adaptation than inventing a new tiny DiT.

[DiT](https://arxiv.org/abs/2212.09748) found that more transformer compute,
including more tokens from smaller patches, consistently improved FID.
[SiT](https://arxiv.org/abs/2401.08740) then improved DiT at every tested model
size without changing model size or FLOPs by moving to continuous-time
interpolants and velocity prediction.  SiT-S itself uses width 384, 12 blocks,
and six heads.  These two independent design lines meet naturally at the local
width-384 U-ViT.

The initial proposal's simultaneous QK normalization, SwiGLU, RoPE, RMSNorm,
and adaLN-zero is rejected for CAP-1.  Those components are credible and some
appear in the official drifting generator, but stacking them with U-ViT skips
would create too many unisolated implementation changes.  CAP-1 uses the
tested U-ViT block: pre-LayerNorm, GELU MLP, a time token, learned positional
embedding, and long skips.  The one small conditioning refinement is the
standard two-layer SiLU time MLP used by DiT/SiT and supported as an option in
U-ViT; it maps the 384-dimensional sinusoidal embedding through width 1,536
and back to 384 before making the time token.  This choice is included in the
parameter count and hardware benchmark.

### 3.3 Why flow matching, not direct one-step regression

For noise $x_0$ and data $x_1$, define

\[
x_t=(1-t)x_0+t x_1,\qquad u_t=x_1-x_0,\qquad t\sim \rho_k.
\]

The foundation minimizes

\[
L_{\mathrm{FM}}(\theta)
=\mathbb E\left[\frac{1}{d}
  \left\|v_\theta(x_t,t)-u_t\right\|_2^2\right],
\qquad d=3\cdot32\cdot32.
\]

Here $\rho_k$ is the timestep law at update $k$.  The original CAP-1 proposal
used the uniform law throughout.  The evidence-vetted CAP-1R protocol uses the
orientation-corrected logit-normal curriculum in Section 18 for the first
60,000 foundation updates, then returns to uniform sampling.  This changes how
finite model capacity and optimizer effort are allocated across time; it does
not change the pointwise conditional velocity target of the linear path.

At its population minimum, squared-loss regression returns the conditional
mean velocity.  Integrating

\[
\frac{dX_t}{dt}=v_\theta(X_t,t),\qquad X_0\sim N(0,I),
\]

transports the source along the learned probability path.  This is the
simulation-free construction developed by
[Flow Matching](https://arxiv.org/abs/2210.02747),
[stochastic interpolants](https://arxiv.org/abs/2303.08797), and
[rectified flow](https://arxiv.org/abs/2209.03003).  SiT's controlled
experiments found that exact finite-interval linear/GVP interpolants and
velocity models outperform the older VP setup.  Linear interpolation is
chosen here because it is nonsingular, matches the repository's successful
B0 implementation, supports a particularly transparent OT coupling, and has
a constant target velocity.

Direct one-step noise-to-image regression is not used: inconsistent random
pairs make the squared-loss optimum a conditional average, the exact collapse
mechanism already seen in this repository.  The ODE retains a time-dependent
path instead.

### 3.4 Why minibatch coupling

Independent noise/data pairs have correct endpoints, but they create crossing
paths and high conditional velocity variance.  [Multisample Flow
Matching](https://arxiv.org/abs/2304.14772) proves that nontrivial minibatch
couplings can preserve the marginals while reducing variance and straightening
flows.  [OT-CFM](https://arxiv.org/abs/2302.00482) gives the corresponding
conditional-flow construction.  More recent work formalizes the expected
minibatch OT plan and its large-batch behavior
([Boite, Delon, and Nadjahi 2026](https://arxiv.org/abs/2605.12174)).

Raw high-dimensional image distance is not an unquestionably correct semantic
cost.  [Optimal Transport Flow Matching by
Design](https://arxiv.org/abs/2606.04092) supplies a particularly relevant
observation: low-frequency image structure can produce substantially
straighter, more useful image flows than arbitrary high-dimensional matching.
CAP-1 therefore uses a fixed multiscale cost rather than a learned encoder.
It is a conservative synthesis of these results, not a theorem copied from
that paper.

For a batch of size $B=40$, define average-pooling maps $P_s$ at scales
$s\in\{1,2,4\}$ and

\[
c_{ij}=\sum_{s\in\{1,2,4\}} a_s
\frac{\|P_s(x_{0,i})-P_s(x_{1,j})\|_2^2}{d_s},
\qquad (a_1,a_2,a_4)=(0.20,0.30,0.50).
\]

An exact linear-sum assignment returns a permutation $\pi$ minimizing
$\sum_i c_{i,\pi(i)}$.  The training pairs are $(x_{0,i},x_{1,\pi(i)})$.
Because the assignment is a permutation, every empirical noise and data point
appears exactly once: the empirical marginals are unchanged.  Gradients do not
pass through the discrete assignment because it constructs regression labels,
not model outputs.

The scale weights are frozen before training.  They emphasize coarse
composition without discarding raw-pixel information.  CAP-1 must log the
raw, pooled-2, and pooled-4 displacement cost relative to independent pairing;
if the implementation does not reduce the declared multiscale cost or fails
the permutation/marginal tests, it is invalid rather than silently falling
back to a different coupling.

## 4. Frozen task and data protocol

### 4.1 Dataset

- Dataset: torchvision CIFAR-10, with archive version and hashes recorded.
- Training target: class id 1, `automobile`, all 5,000 training images.
- Final target: all 1,000 official test automobiles.
- Resolution: native 32x32 RGB; no resize during training.
- Augmentation: horizontal flip with probability 0.5 only.
- No class sweep, sample filtering, aesthetic filtering, or use of test images
  in training or hyperparameter selection.

Automobile is fixed because it is visually coherent, horizontal reflection is
label preserving, and recognizability is much easier to explain than for
ambiguous CIFAR animals.  This is a capability reduction, not a claim that
automobiles are representative of every class.

### 4.2 Invertible conditioning transform

Compute a per-channel mean $\mu$ and one global RMS scale $s>0$ from the 5,000
training automobiles after mapping bytes to $[-1,1]$.  Train in coordinates

\[
T(x)=\frac{x-\mu}{s}.
\]

The same frozen $T$ is applied to noise/data pairing and every training loss;
$T^{-1}$ is used for display and reporting.  This is an invertible affine
coordinate change, not a learned feature encoder.  It aligns the target's
typical second moment with the $N(0,I)$ prior and makes Euclidean kernel and OT
scales interpretable.  Store $\mu$, $s$, their source hash, and the resulting
per-channel moments in the run manifest.

Do not clamp model states during ODE integration.  Standard image quantization
for reporting may clip only after $T^{-1}$; the out-of-range pixel fraction and
clipping magnitude must be reported so clipping cannot hide instability.

### 4.3 Split discipline

The 1,000 test automobiles are sealed until both continuation arms and their
checkpoints are final.  Foundation activation uses only generated samples,
training-set diagnostics, an evaluator whose weights are frozen before CAP-1,
and the prespecified gate in Section 9.  Final evaluation consumes the test
split once for all prespecified checkpoints and NFEs.

Exact train/test pixel duplicates and near-duplicates are audited before the
run.  They are reported, not adaptively deleted.  A generated sample's nearest
training image is always computed against the unaugmented 5,000-image training
set.

## 5. Frozen generator

### 5.1 Architecture

| Component | Frozen value |
|---|---:|
| Input/output | 3 x 32 x 32 standardized pixels / velocity |
| Patch size | 2 |
| Image tokens | 16 x 16 = 256 |
| Extra tokens | one sinusoidal time token after a 384 -> 1,536 -> 384 SiLU MLP |
| Hidden width | 384 |
| Attention heads | 6 (64 dimensions/head) |
| Executed blocks | 6 input + 1 middle + 6 output = 13 |
| MLP | ratio 4, GELU |
| Normalization | pre-LayerNorm |
| Long skips | concatenate mirrored input/output states, then linear 768 -> 384 |
| Position | learned 2-D token-position parameter, initialized N(0, 0.02) |
| Decoder | linear unpatchify followed by one 3x3 RGB convolution |
| Dropout | none |
| Class conditioning / CFG | none; the target has one class |
| Parameter count | approximately 26.1M; exact count is a preflight assertion |

The implementation should follow the official
[`UViT`](https://github.com/baofff/U-ViT/blob/main/libs/uvit.py) topology, with
width, the enabled time-token MLP, loss target, and pixel standardization
changed.  “Depth 12” in its configuration executes 13 blocks because there
are six blocks on each side of one middle block; CAP-1 records the executed
count to avoid this ambiguity.

### 5.2 Numerical implementation

- PyTorch `2.7.1+cu126`, torchvision `0.22.1+cu126`, Python 3.12.
- CUDA scaled-dot-product attention.
- Fused AdamW where supported.
- TF32 enabled for float32 matrix multiplies.
- FP16 autocast with dynamic gradient scaling is the default.
- A BF16 alternative may replace FP16 only if a pre-training numerical and
  throughput preflight shows finite matching losses and at least equal speed;
  this is an implementation decision, not an output-quality selection.
- Eager mode is the authority.  `torch.compile` may be used only if 100 fixed
  forward/backward updates agree with eager within declared precision and it
  improves median throughput by at least 10% after compilation warmup.
- No activation checkpointing in ordinary flow updates.  Block checkpointing
  is enabled only for the differentiable auxiliary trajectory if needed to
  remain below 90% of physical GPU memory.
- Atomic checkpoints include model, optimizer, EMA, scaler, update number,
  data cursor, and all RNG states.

The final implementation must not inherit the ambient CPU-only `uv` PyTorch
wheel.  The CUDA package source and `torch.cuda.is_available()` are hard
preflight checks.

## 6. Optimization and time budget

### 6.1 Foundation

| Item | Frozen value |
|---|---:|
| Batch size | 40 |
| Updates | 100,000 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Betas | (0.99, 0.999) |
| Weight decay | 0.03 |
| Warmup | linear, 2,500 updates |
| Schedule after warmup | constant |
| EMA | 0.9999, FP32 state |
| Time sampling | updates 0--59,999: $\operatorname{logit}(t)\sim N(-0.8,1)$; thereafter uniform |
| Loss reduction | mean over batch and pixel dimensions |
| Ordinary gradient clipping | none; fail on persistent nonfinite values |
| Checkpoints | 0, 20k, 40k, 60k, 80k, 100k |

This is close to the official U-ViT CIFAR optimizer, with its learning rate
halved for the smaller local batch.  Constant $10^{-4}$ and EMA 0.9999 are also
the official SiT defaults.  The choices are fixed rather than tuned on the
one-class test set.  The foundation presents $100{,}000\cdot40/5{,}000=800$
nominal dataset passes and a completed continuation presents 1,000.  For
context, the official U-ViT CIFAR schedule presents roughly 1,280 nominal
passes ($500{,}000\cdot128/50{,}000$).  CAP-1 is therefore shorter in data
exposure as well as vastly smaller in aggregate compute; the one-class task is
what makes that reduction plausible.

### 6.2 Forked continuations

Both continuations start from the exact 100k raw model, optimizer, and EMA
state.  Both receive the same next 25,000 flow batches, OT assignments,
timesteps, and endpoint noises.  Both use a constant learning rate of `5e-5`.
The only causal difference is the candidate's auxiliary update.

- `control`: 25,000 more flow-only updates;
- `protected_geometry`: the same 25,000 flow updates, with one shared
  four-Euler-step differentiable generated trajectory every tenth update and
  the protected B1/B2 correction in Section 8.

Continuation order on the physical GPU must not affect streams.  Clone the
fork artifact, run arms into separate directories, and restore all states
before starting the second arm.

### 6.3 Measured local feasibility

A synthetic faithful-topology benchmark was run on the actual RTX 4050 Laptop
GPU before writing this protocol.  For width 384, patch 2, 13 executed blocks,
FP16, fused AdamW, SDPA, no block checkpointing, and batch 40, it measured:

```text
parameters             26.12 M
median step             approximately 0.387 s
throughput              approximately 103 images/s
peak allocated memory   approximately 2.25 GiB
```

This is a synthetic upper-bound estimate, not an end-to-end training promise.
At that rate, 100k foundation updates take about 10.8 hours and each pure
25k continuation about 2.7 hours.  Allowing 30--50% overhead in the candidate,
data loading, checkpoints, and evaluation gives an expected 18--21 hour job
with a 3--6 hour safety margin.

The executable preflight replaces this estimate with measurements from 500
real data/OT updates and 20 correction events.  Training must reserve the last
two wall-clock hours for final evaluation.  If projected total time exceeds 24
hours, reduce continuation updates equally before changing architecture,
batch size, patch size, or the foundation.  Patch size 2 already fits and is
not an outcome-dependent tuning knob.

## 7. Sampling

The primary sampler is deterministic Heun integration from $t=0$ to $t=1$.
Report at fixed total network-evaluation budgets:

- NFE 1: one Euler step, diagnostic only;
- NFE 8: four Heun steps;
- NFE 32: sixteen Heun steps, primary quality readout.

Also generate a fixed-latent **NFE-64 secondary audit** with 32 Heun steps.
It is a sampler-resolution diagnostic, not a checkpoint-selection metric and
not a replacement for the predeclared NFE-32 primary comparison.  A gain at 64
would show that numerical integration, rather than the learned field alone, is
limiting visible quality; no gain would justify retaining the cheaper sampler.

All arms use the same latent tensors at every NFE.  States are not clamped.
The ODE direction, endpoint, and NFE accounting require unit tests: for a
constant velocity field, Euler and Heun must land at the analytic endpoint,
and “NFE 8” must invoke the network exactly eight times.

SiT reports that stochastic sampling can improve final FID at high NFE, but
correct stochastic sampling requires a score conversion and a diffusion
coefficient tied to the chosen interpolant.  CAP-1 does not improvise that
machinery.  A stochastic readout is a later experiment after the deterministic
model is known to work.

## 8. Protected formalization-derived continuation

### 8.1 B1: global spectral protection

For fixed frequencies $\omega_r$, B1 estimates

\[
L_{\mathrm{B1}}
=\frac1M\sum_{r=1}^{M}
\left|\widehat\varphi_p(\omega_r)-
      \widehat\varphi_q(\omega_r)\right|^2.
\]

At population level, equality of characteristic functions determines a
probability law.  Finite random features do not prove equality, but the local
B1 experiment showed that this term protected global geometry and gave the
best matched historical KID/FID.  CAP-1 reuses the audited implementation with
256 training frequencies, its coarse-to-fine full-support frequency mixture,
a 512-feature held-out audit bank, and a 25-event 25% refresh schedule.  The
frequency scale is recalibrated from the standardized automobile training
data; no old CIFAR-wide coefficient is reused.

### 8.2 B2: raw identifiable term plus balanced multiscale term

For probes $z$, positives $Y\sim p$, generated negatives $X\sim q$, and the
Laplace kernel $k_\tau(a,b)=\exp(-\|a-b\|_2/\tau)$, define the row-normalized
empirical field

\[
\widehat V_\tau(z)=
\frac{\sum_j k_\tau(z,Y_j)Y_j}{\sum_j k_\tau(z,Y_j)}-
\frac{\sum_j k_\tau(z,X_j)X_j}{\sum_j k_\tau(z,X_j)}.
\]

The raw full-resolution term is

\[
L_{\mathrm{raw}}=\frac1{|Z|d}
\sum_{z\in Z}\|\widehat V_\tau(z)\|_2^2.
\]

This is the closest finite surrogate to the project's arbitrary-target
Laplace converse.  Probes are independently drawn target images plus Gaussian
noise.  They are not clipped, so the ideal probe law has full Euclidean
support.  Positives, probe centers, and generated negatives use independent
streams.  Exact Laplace weights are evaluated as stable softmaxes.

Historical B2 showed that one-sided normalization can lower drift while
compressing rank.  [Sinkhorn Drifting](https://arxiv.org/abs/2603.12366)
explains why: two-sided scaling introduces the missing marginal balance and
reports improved temperature robustness and one-step quality.  CAP-1 therefore
also forms positive and negative kernel matrices at resolutions 32, 16, and 8,
applies exactly three alternating row/column Sinkhorn normalization rounds,
and computes the corresponding balanced-centroid field energy at each scale.

The frozen discrepancy is

\[
L_{\mathrm{B2}}
=0.25 L_{\mathrm{raw},32}
+0.75\left(0.20L_{\mathrm{bal},32}
           +0.30L_{\mathrm{bal},16}
           +0.50L_{\mathrm{bal},8}\right).
\]

Each energy is divided by its coordinate dimension.  Each scale gets a
separate target-only bandwidth calibrated to median self-excluded ESS fraction
0.60.  Log median, 5th percentile, and minimum ESS; maximum row weight; row and
column mass errors before and after balancing; and all three field energies.
No bandwidth is selected from generated quality.

The positive coefficient on $L_{\mathrm{raw},32}$ matters.  If the sum of
nonnegative population analogues were exactly zero, the raw identifiable term
would also be zero.  The balanced and lower-resolution terms are optimization
aids; they do not replace the certified full-resolution zero-set connection.

### 8.3 One shared correction trajectory

Every tenth candidate update:

1. draw 32 new Gaussian priors from an auxiliary-only stream;
2. generate 32 samples with four differentiable Euler steps;
3. use that same generated batch for B1 and B2;
4. draw independent B1 targets, B2 positives, and B2 probe centers;
5. compute component gradients before the ordinary optimizer update.

Sharing only the generated trajectory removes almost half the expected
correction cost.  It does not reuse target roles or make the discrepancy
estimators identical.  The control receives no substitute auxiliary update;
its fairness is equal flow data and flow updates, while compute overhead is
reported separately.

### 8.4 Gradient calibration and conflict protection

Old numerical coefficients are meaningless for a new 26M model and new
normalization.  Before the forked run, use 16 calibration events drawn only
from training data to freeze $\lambda_1$ and $\lambda_2$ such that median
unprojected event-gradient ratios are

\[
\frac{\|\lambda_1\nabla L_{\mathrm{B1}}\|}{
      \|\nabla L_{\mathrm{FM}}\|}=0.20,
\qquad
\frac{\|\lambda_2\nabla L_{\mathrm{B2}}\|}{
      \|\nabla L_{\mathrm{FM}}\|}=0.10.
\]

Freeze these coefficients before either continuation.  On each correction
event, if an auxiliary gradient $g_a$ conflicts with the flow gradient $g_0$,
replace it with

\[
g_a^\perp=g_a-
\frac{\langle g_a,g_0\rangle}{\|g_0\|^2}g_0
\quad\text{when }\langle g_a,g_0\rangle<0,
\]

and otherwise leave it unchanged.  This is the one-primary-task form of
[PCGrad](https://arxiv.org/abs/2001.06782).  It guarantees
$\langle g_a^\perp,g_0\rangle\ge0$, so the auxiliary cannot oppose the flow
loss to first order.  If the norm of the combined projected auxiliary exceeds
$0.25\|g_0\|$, rescale the auxiliary sum to that cap.  Then apply

\[
g_{\mathrm{update}}=g_0+g_{\mathrm{B1}}^\perp+
g_{\mathrm{B2}}^\perp.
\]

This protection addresses the exact failure observed locally: a useful drift
signal was allowed to compress a working model's rank.  It does not prove that
the nonconvex update will improve sample quality; it preserves the primary
descent direction only locally.  Log component norms, pairwise cosines,
projection frequency, cap frequency, and actual post-projection ratios.

The mixed-precision implementation has one non-negotiable detail.  Obtain the
three component gradients separately from individually AMP-scaled losses,
divide the out-of-place `autograd.grad` results by the one common current AMP
scale, cast each tensor to float32, and only then form global dot products,
norms, projections, and the cap.  After constructing the unscaled float32
$g_{\mathrm{update}}$, multiply it by that same scale when assigning `.grad`;
then call `GradScaler.step` exactly once so PyTorch performs its ordinary one
unscale and nonfinite check.  Calling `unscale_` first *and* storing an
unscaled update for `GradScaler.step` would divide twice and is forbidden.  Do
not project loss-scaled or float16 gradients.  PyTorch's AMP documentation
explicitly requires gradients to be unscaled before inspection or
modification.  A preflight test must compare this implementation to a float32
reference on a tiny model, including a deliberately conflicting auxiliary,
and must verify the reported dot products and the actual parameter update.

## 9. Preflight and activation gate

### 9.1 Mandatory implementation preflight

Before the long run:

1. verify dataset counts, class id, pixel ranges, transform invertibility, and
   train/test hashes;
2. verify the OT assignment is a permutation and preserves both empirical
   marginals exactly;
3. verify analytic constant-velocity Euler/Heun endpoint tests and NFE counts;
4. verify eager mixed-precision forward/backward agreement against FP32 on a
   fixed mini-batch;
5. overfit 64 fixed noise/data pairs over a fixed time grid until velocity MSE
   falls by at least 95%; this catches sign, time, and unpatchify errors;
6. run 500 real foundation updates and 20 candidate events, requiring finite
   losses, finite gradients, valid ESS/mass diagnostics, and peak physical
   memory below 90%;
7. measure projected wall time from synchronized steady-state steps;
8. serialize, resume, and demonstrate bitwise-equal next data/noise streams and
   numerically equal next loss.

No image-quality result from preflight may change architecture or loss
hyperparameters.  Failures repair correctness or feasibility and rerun the
preflight.

### 9.2 Foundation activation at 100k

Generate 1,024 fixed-latent NFE-32 samples from the 100k EMA without using the
official test images.  Activate the fork only if all of the following hold:

- no nonfinite values;
- post-inverse-transform clipping affects at most 5% of scalar pixels;
- a frozen CIFAR-10 evaluator assigns at least 50% top-1 `automobile` and at
  least 75% combined `automobile`/`truck` probability mass on average;
- duplicate rate under exact 8-bit hashes is below 1%;
- at least 512 distinct training images are nearest neighbors of the 1,024
  generated images in the declared fixed reporting feature space;
- generated raw-pixel effective rank is at least half the same-size
  training-resample rank;
- the uncurated fixed 8x8 grid is saved.

The classifier thresholds are an automatic recognizability screen, not a
claim that classifier confidence equals human realism.  The grid and
memorization controls travel with the verdict.  The distinct-neighbor veto is
deliberately loose: 1,024 uniform draws over 5,000 equally occupied training
cells would visit about
$5{,}000\left(1-(4{,}999/5{,}000)^{1{,}024}\right)\approx927$ cells in
expectation, whereas 512 permits substantial nonuniformity but rejects a
small-bank memorizer.

If this gate fails, B1/B2 is not allowed to “repair” a nonworking foundation.
Spend the continuation budget on 50,000 additional foundation updates and
evaluate the resulting 150k foundation as a capability failure or success.
This fallback maximizes the chance of leaving the day with recognizable images
without interpreting an auxiliary on an invalid base model.

## 10. Final evaluation

### 10.1 Frozen sample sets

For the 100k foundation, 125k control, and 125k protected arm, generate:

- the same 1,000 latents at NFE 1, 8, 32, and the secondary NFE-64 audit;
- an additional 4,000 NFE-32 samples for train-reference and memorization
  diagnostics if time permits;
- fixed uncurated 8x8 grids and identical-latent side-by-side triptychs;
- trajectory panels at $t=0,.25,.5,.75,1$ for 16 fixed latents.

Never select or reorder samples by a quality score.  Any hand-selected image
must be labeled separately and cannot replace the fixed grid.

### 10.2 Primary and secondary measurements

The primary statistical endpoint is **CIFAR-feature KID**, because its MMD
U-statistic is unbiased at finite sample size, unlike plug-in FID.  See
[Demystifying MMD GANs](https://openreview.net/forum?id=r1lUOzWCW).

Report:

1. CIFAR-feature KID against all 1,000 held-out automobiles, using the unbiased
   order-2 MMD estimator and the fixed cubic kernel
   $k(a,b)=(a^\top b/64+1)^3$, plus 1,000 paired bootstrap contrasts;
2. target-class hit rate and confidence from a frozen CIFAR-10 classifier;
3. improved precision, recall, density, and coverage in the fixed reporting
   feature space;
4. CIFAR-feature Fréchet distance as a descriptive small-sample statistic
   only, with the number of real and generated samples in the table heading;
5. raw-pixel effective rank, per-frequency-band DCT variance/effective rank,
   radial spectrum slope, color moments, and clipping rate (a full orthonormal
   DCT preserves the global covariance spectrum and is not counted as an
   independent rank metric);
6. exact duplicate rate and nearest-training-image distances in pixels, DCT,
   and a reporting-only perceptual feature space;
7. held-out B1, raw B2, and balanced multiscale B2 discrepancies using banks,
   probes, positives, and priors not used in auxiliary training;
8. wall time, peak physical and allocated GPU memory, training forward NFE,
   correction NFE, and sampling NFE.

FID's finite-sample and model-dependent bias is documented by
[Chong and Forsyth](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html).
Precision and recall must be reported together because either alone rewards a
known failure mode; the standard manifold estimator is described by
[Kynkaanniemi et al.](https://arxiv.org/abs/1904.06991), and density/coverage by
[Naeem et al.](https://proceedings.mlr.press/v119/naeem20a.html).

Use the externally pretrained `cifar10_resnet56` checkpoint from
[`chenyaofo/pytorch-cifar-models`](https://github.com/chenyaofo/pytorch-cifar-models)
at repository revision `786c16252c0fc58ee9adac063f8337cc4a7a497a` as the
prespecified evaluator.  The repository reports 94.37% CIFAR-10 top-1
accuracy.  Its [archived training
log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/resnet56/default.log)
specifies input mean
`(0.4914, 0.4822, 0.4465)` and standard deviation
`(0.2023, 0.1994, 0.2010)` after mapping bytes to `[0,1]`; use those exact
values.  Save the downloaded weight-file SHA-256 and a hash of the model
source; fail instead of silently substituting a different checkpoint.  Class
recognizability uses its logits, while KID, FID,
precision/recall, density/coverage, and perceptual nearest-neighbor checks use
its 64-dimensional penultimate pooled feature without fitted rescaling or
normalization.  This CIFAR-native representation is more
appropriate for 32 x 32 images than resizing them for an ImageNet classifier,
but it is still only one learned evaluator, so all conclusions must be
cross-checked against pixel/DCT statistics and uncurated grids.

The evaluator is fixed before CAP-1 and never fine-tuned on generated images.
It is not allowed to appear in B1, B2, OT costs, checkpoint selection, or any
generator update.  Its full ten-class test accuracy and automobile recall are
recomputed once in preflight to catch preprocessing mistakes; a full-test
top-1 below 94.0% is a hard preprocessing/checkpoint failure.

### 10.3 Floors and uncertainty

Using only the 1,000 test automobiles, repeatedly split 500/500 without
replacement and report real-versus-real distributions for every set metric.
Also compare a held-out test half with a training half to expose ordinary
train/test shift.  Candidate/control contrasts use common latent indices and
paired bootstrap resampling: every replicate resamples 1,000 real indices and
1,000 latent indices with replacement, and applies the same index draws to
both arms.  The point estimate is the all-sample U-statistic; bootstrap
replicates estimate uncertainty of the paired arm contrast.  Cross-checkpoint intervals quantify sample
uncertainty, not training-seed uncertainty.

Do not quote the paper's 50,000-sample ImageNet FID beside CAP-1's
class-conditional 1,000-sample FID as though they were comparable.

## 11. Decision rules

### 11.1 Capability verdict

The experiment demonstrates the intended limited capability only if the final
selected prespecified arm:

- passes the recognizability, duplication, rank, and clipping conditions in
  Section 9.2 on held-out data as well as the training-only gate;
- has KID and FID clearly separated from the Gaussian-noise control in the
  correct direction;
- shows multiple recognizable, nonduplicate automobiles in the uncurated
  grid and across latent trajectories;
- has nontrivial precision and recall rather than high precision with near-zero
  recall;
- passes the nearest-training-image audit.

The “selected arm” is the protected arm only if it satisfies Section 11.2;
otherwise it is the pure-flow control.  This ordering is declared before test
evaluation.

### 11.2 Geometry-correction promotion

Promote `protected_geometry` over `control` only if all are true at NFE 32:

1. held-out KID improves by at least 5%, and at least 90% of paired bootstrap
   resamples favor the candidate;
2. automobile top-1 rate is no more than two percentage points worse;
3. recall is no more than 0.03 worse and precision is no more than 0.03 worse;
4. raw-pixel effective-rank ratio candidate/control is at least 0.90;
5. held-out raw B2 energy falls by at least 15%;
6. duplicate and memorization vetoes pass;
7. all costs are reported.

These are pilot promotion rules, not hypothesis-test significance thresholds.
A candidate that greatly improves drift but fails rank or image quality repeats
the historical B2 mechanism result and is not the flagship.

## 12. Artifact and audit requirements

Create a dedicated package and run directory rather than modifying consumed
B0/B1/B2/B3 artifacts.  Before the long run, freeze:

- source manifest and git commit;
- full config JSON and SHA-256 sidecar;
- dataset archive, split, and transform hashes;
- evaluator identity and hash;
- architecture parameter count and module summary;
- hardware, driver, CUDA, PyTorch, and dependency versions;
- seed derivation and independent stream labels;
- measured preflight speed/memory projection;
- B1/B2 bandwidth and gradient-calibration artifacts.

During training, append machine-readable JSONL containing losses, learning
rate, scaler, throughput, memory, EMA norm, OT costs, gradient norms/cosines,
projection/cap events, ESS quantiles, Sinkhorn mass errors, and checkpoint
hashes.  On resume, never overwrite a final artifact; verify the last atomic
checkpoint and RNG/data cursors first.

Tests must cover:

- time and ODE orientation;
- patchify/unpatchify round trip and output shape;
- exact OT permutation/marginals and deterministic assignment;
- B1/B2 role independence;
- Sinkhorn positivity and improving row/column mass errors;
- finite loss and nonzero model gradients for every term;
- PCGrad dot-product and norm-cap invariants;
- common-random-number equality of the control/candidate flow streams;
- checkpoint/resume reproducibility;
- evaluator isolation from training.

## 13. Rejected alternatives and why

### Exact official drifting model

The official [drifting code](https://github.com/lambertae/drifting) uses a
133M-parameter B/2 generator even for the ablation, pretrained MAE features,
large memory banks, multiple feature stages and temperatures, class/noise
conditioning, and 64 TPU v6e devices for its 30k-step ablation.  It is the
right eventual baseline but not a one-day 6 GB local experiment.  CAP-1 must
not call its much smaller correction a paper reproduction.

### External pretrained latent autoencoder

This would reduce transformer tokens and could improve throughput, but it
would make the generator's support and reconstruction ceiling depend on an
external learned encoder/decoder precisely when this program is testing
encoder-free training.  Pixel-space 32x32 already fits.

### Patch size 4

It is faster, but it removes 75% of the image tokens.  DiT scaling and U-ViT's
CIFAR configuration favor patch 2, and the measured patch-2 model fits with a
safe margin.

### Patch Diffusion

[Patch Diffusion](https://arxiv.org/abs/2304.12526) is compelling: it reports
at least 2x training speed and improved training from as few as 5,000 images.
But it changes the training input, adds patch-location/size conditioning, and
has no direct validation here with a U-ViT flow objective at 32x32.  It is the
first capacity-efficiency follow-up if CAP-1 underfits, not an additional risk
in the first transformer run.

### U-shaped timestep sampling and perceptual Huber loss

[Improved Rectified Flow](https://arxiv.org/abs/2405.20320) obtains strong
low-NFE gains from these choices in a reflow setting.  The perceptual loss
reintroduces an external encoder and the U-shaped law is motivated for reflow,
not this first flow.  Neither is imported blindly.  This rejection concerns
that reflow-specific U-shaped law; it does not reject the separately sourced
logit-normal-to-uniform curriculum adopted in Section 18.

### Learned guidance or classifier-free guidance

There is one class and no null-vs-class distinction worth guiding.  Guidance
would add an extra training condition and can trade diversity for fidelity.
It is excluded.

### Training the full day with B1/B2 from initialization

This would make an unrecognizable result ambiguous between generator failure
and correction failure.  The shared-foundation fork makes the correction
causal and guarantees that a working foundation is preserved.

## 14. Expected interpretations

| Outcome | Interpretation | Next move |
|---|---|---|
| Foundation fails | One-class/pixel U-ViT is still undertrained or incorrectly optimized | Inspect preflight/logs; try Patch Diffusion or a longer/cloud run before drift |
| Control works, protected arm fails quality/rank | B1/B2 still conflicts beyond first-order protection | Keep control; inspect scale-specific gradients and do not promote drift |
| Both work, protected lowers drift only | Formal mechanism is active but not the image-quality bottleneck | Report mechanism separately; improve perceptual-free geometry before retrying |
| Protected arm improves KID/visuals and passes rank | First credible encoder-free image-quality gain from the formalization-derived correction | Repeat across at least three training seeds and a second class |
| Both are strong and nearly tied | Generator capacity/path design was the dominant missing ingredient | Promote pure flow as capability result; treat correction as neutral |

## 15. Implementation order

1. Create the isolated CAP-1 package, immutable config schema, stream registry,
   and artifact helpers.
2. Implement/test the affine data transform and exact multiscale BatchOT.
3. Implement the faithful width-384 U-ViT and analytic ODE tests.
4. Implement flow training, EMA, atomic resume, and fixed-latent sampling.
5. Run the 64-pair overfit and 500-update real-data preflight.
6. Port B1 and raw B2 without changing their audited mathematical roles.
7. Add three-round balanced multiscale B2 and its mass/ESS tests.
8. Add shared correction trajectories, fixed gradient calibration, PCGrad, and
   component logging.
9. Run the correction memory/time preflight and freeze the manifest.
10. Train the 100k foundation and apply the automatic activation gate.
11. If it passes, clone and run the two 25k continuations; otherwise run the
    declared 50k foundation fallback.
12. Freeze checkpoints, consume the test split once, generate every declared
    sample set, compute metrics/floors/bootstraps, and write an honest result
    report.

## 16. Literature ledger

The design depends on the following primary sources; the description above
states exactly what is borrowed and what remains our synthesis.

- Deng et al., [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770)
  and [official code](https://github.com/lambertae/drifting): motivating model,
  bi-normalized multitemperature drift, and scale of the real baseline.
- Bao et al., [All are Worth Words: A ViT Backbone for Diffusion
  Models](https://arxiv.org/abs/2209.12152): U-shaped transformer skips,
  time/image tokens, patch-2 CIFAR model, and convolutional output head.
- Peebles and Xie, [Scalable Diffusion Models with
  Transformers](https://arxiv.org/abs/2212.09748): transformer and patch-size
  scaling.
- Ma et al., [SiT](https://arxiv.org/abs/2401.08740): continuous-time velocity
  prediction, exact interpolants, width-384 small transformer, EMA, and
  deterministic/stochastic sampling distinction.
- Lipman et al., [Flow Matching for Generative
  Modeling](https://arxiv.org/abs/2210.02747), Albergo and Vanden-Eijnden,
  [Stochastic Interpolants](https://arxiv.org/abs/2303.08797), and Liu et al.,
  [Flow Straight and Fast](https://arxiv.org/abs/2209.03003): population path
  and regression foundations.
- Pooladian et al., [Multisample Flow
  Matching](https://arxiv.org/abs/2304.14772), Tong et al.,
  [OT-CFM](https://arxiv.org/abs/2302.00482), and Boite et al.,
  [Expected Batch OT Plans](https://arxiv.org/abs/2605.12174): correct-marginal
  minibatch coupling, variance, and path straightness.
- Malnick et al., [Optimal Transport Flow Matching by
  Design](https://arxiv.org/abs/2606.04092): low-frequency structure as useful
  image transport geometry.
- He et al., [Sinkhorn-Drifting Generative
  Models](https://arxiv.org/abs/2603.12366): two-sided balance, definiteness,
  temperature robustness, and training-cost tradeoff.
- Yu et al., [Gradient Surgery for Multi-Task
  Learning](https://arxiv.org/abs/2001.06782): projecting conflicting
  auxiliary gradients.
- [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html):
  gradients must be unscaled before norm measurement or in-place modification.
- Chen Yaofo's [pretrained CIFAR
  models](https://github.com/chenyaofo/pytorch-cifar-models): the pinned
  CIFAR-native ResNet-56 reporting model and published accuracy.
- Binkowski et al., [Demystifying MMD
  GANs](https://openreview.net/forum?id=r1lUOzWCW), Chong and Forsyth,
  [Effectively Unbiased
  FID](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html),
  Kynkaanniemi et al., [Improved Precision and
  Recall](https://arxiv.org/abs/1904.06991), and Naeem et al.,
  [Reliable Fidelity and Diversity
  Metrics](https://proceedings.mlr.press/v119/naeem20a.html): evaluation and
  finite-sample limitations.

## 17. Bottom line

The most effective experiment is not “put a transformer under the old loss
and hope.”  It is a layered risk reduction:

1. a published pixel-CIFAR transformer topology;
2. a theoretically valid, variance-reducing continuous-time transport
   objective;
3. a measured architecture that fits the real 6 GB device;
4. a shared foundation that must visibly work before experimental geometry is
   allowed to act;
5. a paired fork in which B1 protects global structure, balanced B2 supplies
   the formalization-derived local signal, and gradient surgery prevents that
   signal from opposing the proven flow objective to first order;
6. an evaluation that cannot hide blur, collapse, memorization, or small-sample
   FID bias behind one attractive number.

This gives CAP-1R the best chance of producing recognizable images within one
day while ensuring that, whatever happens, the result answers a precise and
useful question.

## 18. Additional mechanism-selection audit

This section records the follow-up investigation prompted by two newer
pixel-space papers:

- Hoogeboom et al., [Simpler Diffusion
  (SiD2)](https://arxiv.org/html/2410.19324v1);
- Lei et al., [There Is No VAE
  (EPG)](https://arxiv.org/html/2510.12586v2).

The search was expanded to official implementations and newer work on
pixel-space flow, prediction parameterization, and timestep allocation.  The
question was deliberately narrow: **which ideas can be transferred into this
one-day, 6 GB, encoder-independent experiment without turning it into an
untested mixture of papers?**

### 18.1 Updated decision

The best runnable mechanism is still the single-stage pixel U-ViT with linear
flow matching and the protected B1/B2 fork.  The new evidence does not justify
replacing it with EPG, a full PixelFlow cascade, or clean-image prediction.
It does justify one low-surface-area training improvement:

> Use a two-phase timestep curriculum for the foundation: a noise-side
> logit-normal distribution for the first 60,000 updates, followed by uniform
> timesteps for all remaining foundation and continuation updates.

This revised protocol is called **CAP-1R** below.  CAP-1R changes neither the
probability path, regression target, network, sampler, data, nor B1/B2
comparison.  It changes only how the fixed training budget is allocated over
the same time interval.  Section 18.7 is authoritative over the time-sampling
row in Section 6.1.  All other earlier frozen choices remain authoritative.

There is an important honesty boundary.  No published experiment exactly
matches all of: one CIFAR class, 5,000 images, a 26M U-ViT, batch 40, linear
flow, minibatch OT, one RTX 4050, and 100k updates.  Consequently, no source
can guarantee recognizable output.  CAP-1R is the strongest evidence-backed
composition under those constraints, not a reproduced literature result.

### 18.2 What the strongest references actually establish

| Source | Direct evidence | Transfer strength for CAP-1R | Decision |
|---|---|---:|---|
| [Meta Flow Matching guide and official image code](https://github.com/facebookresearch/flow_matching/tree/main/examples/image) | Pixel-space continuous flow on CIFAR-10 reaches reported FID 2.07 with an unconditional U-Net, EMA, skewed timesteps, and a 50-NFE second-order sampler | High for the flow mechanism; medium for our transformer and budget | Use as the implementation authority for path/sampler tests, not as a transformer claim |
| [Official U-ViT CIFAR configuration](https://github.com/baofff/U-ViT/blob/main/configs/cifar10_uvit_small.py) | Native 32x32 pixels, patch 2, width 512, 12 configured layers, batch 128, 500k updates | High for topology; indirect for linear flow | Keep the U-shaped long skips, patch 2, and convolutional output head |
| [Official SiT implementation](https://github.com/willisma/SiT) | Continuous-time interpolants, linear paths, velocity prediction, AdamW at 1e-4, and EMA work with transformer generators | High for the objective; indirect because published models operate in VAE latents and at much larger scale | Keep linear velocity flow; do not call CAP-1 a SiT reproduction |
| [Curriculum Sampling](https://arxiv.org/html/2603.12517) | On 32x32 pixel CIFAR CFM, logit-normal then uniform improves the reported best FID from 3.85 at 150k to 3.22 at 100k; the winning trace switches at 60k | High task/objective match, but medium overall evidence because it is one workshop study using a 55M U-Net and batch 1,024 | Adopt the published 60k switch and correct its time orientation |
| [SiD2](https://arxiv.org/html/2410.19324v1) | Strong high-resolution pixel diffusion from sigmoid loss weighting, token-heavy scaling, and a simplified residual U-ViT | High evidence that pixel-space generation can work; low direct transfer to 32x32 linear flow | Keep small patches and skips; do not transplant its diffusion loss |
| [PixelFlow](https://arxiv.org/html/2504.07963) and [official code](https://github.com/ShoufaChen/PixelFlow) | A multiresolution pixel-flow cascade reaches reported ImageNet-256 FID 1.98; at target 64, a moderate low-resolution kickoff preserves quality and slightly improves FID | Medium: same pixel/flow family, but a different path, XL model, 600k-step low-resolution ablation, and stage-aware inference | First architectural follow-up if CAP-1R is compute-limited, not part of the first run |
| [JiT](https://arxiv.org/html/2511.13720) and [official code](https://github.com/LTH14/JiT) | Direct clean-image prediction prevents transformer failure when patch dimension approaches or exceeds hidden width | High at aggressive high-resolution patching; low need at patch 2 and width 384 | Retain direct velocity prediction for CAP-1R |
| [EPG](https://arxiv.org/html/2510.12586v2) and [official code](https://github.com/AMAP-ML/EPG) | Internal encoder pretraining followed by end-to-end encoder/decoder generative training gives strong high-resolution pixel models without an external VAE | High large-scale evidence, but incompatible with a single-stage one-day run | Reserve for a future multi-GPU program that permits a separate internal pretraining phase |

The Meta result is especially useful for calibration.  It confirms that the
linear-flow family is capable of excellent pixel-CIFAR generation, but the
reported model is a U-Net trained for roughly 1,800 effective epochs.  It
therefore validates the mechanism, not the claim that our smaller transformer
must converge in 100k updates.  Conversely, official U-ViT validates the
transformer topology but uses diffusion/noise prediction and 500k updates.
CAP-1R necessarily bridges those two evidence lines.

### 18.3 Why the timestep curriculum transfers cleanly

At each fixed time $t$, conditional flow matching performs a squared-loss
regression whose unrestricted population solution is

\[
v^*(x,t)=\mathbb E[x_1-x_0\mid x_t=x].
\]

Replacing uniform time sampling by any density $\rho(t)>0$ almost everywhere
does not change this pointwise conditional target.  It changes how a
finite-capacity, shared-parameter network allocates approximation effort over
time.  This is precisely the effect wanted from a curriculum.  We do **not**
importance-correct the loss by $1/\rho(t)$: that would largely undo the
curriculum and amplify variance in rarely sampled regions.

The published curriculum uses

\[
z_s=(1-s)x_{\mathrm{data}}+s\epsilon,
\qquad \operatorname{logit}(s)\sim N(0.8,1),
\]

so larger $s$ is the noise side.  CAP-1R uses the opposite coordinate,

\[
x_t=(1-t)x_{\mathrm{noise}}+t x_{\mathrm{data}}.
\]

Setting $t=1-s$ gives

\[
\operatorname{logit}(t)
=-\operatorname{logit}(s)
\sim N(-0.8,1).
\]

The sign reversal is mandatory.  Copying $+0.8$ directly would emphasize the
data side rather than reproduce the paper's successful noise-side structure
phase.

The switch is kept at the paper's actual 60k update, rather than rescaled to
40k merely because the authors call 60k roughly 40% of their nominal 150k
training budget.  Their winning checkpoint is itself at 100k: it therefore
contains 60k curriculum updates followed by 40k uniform updates, exactly the
foundation horizon available here.  This is the least interpretive transfer.
It also has a useful safety property: the final 40% of the foundation and both
continuations restore full uniform coverage, avoiding the ceiling observed for
a permanently middle-biased sampler.

### 18.4 Why SiD2 is not copied as a flow loss

SiD2's central optimization result is a shifted sigmoid weight in diffusion
log-SNR coordinates.  It is not simply a better generic MSE.  For CAP-1R's
linear path, define

\[
\lambda(t)=2\log\frac{t}{1-t},
\qquad
\widehat x_1=x_t+(1-t)\widehat v.
\]

Then

\[
\|\widehat x_1-x_1\|^2
=(1-t)^2\|\widehat v-v\|^2,
\qquad
|\lambda'(t)|=\frac{2}{t(1-t)}.
\]

Consequently, a SiD2-style $x$-space sigmoid weighting would induce, up to a
constant, the velocity weight

\[
w_b^{(v)}(t)
=\frac{2(1-t)}{t}\,
  \sigma\!\left(\lambda(t)-b\right).
\]

This is mathematically implementable and remains positive on the interior,
but it is not calibrated for native 32x32 linear flow.  SiD2 reports
resolution-dependent shifts only at substantially larger resolutions, and
the factor above can be numerically severe near an endpoint.  Adding it now
would confound time sampling, loss weighting, and the B1/B2 intervention.
It is therefore recorded as a later one-factor ablation, not activated in
CAP-1R.

SiD2 nevertheless reinforces three current choices:

1. spending compute on more image tokens can outperform spending the same
   compute only on channel width;
2. a separate Haar multiscale reconstruction loss is not automatically
   helpful once noise/loss weighting is calibrated;
3. removing long/blockwise skips slightly hurts the smaller models, so the
   first local 26M model should retain U-ViT skips.

### 18.5 Why clean-image prediction is not activated at 32x32

JiT gives an important scaling rule, not a universal instruction to replace
velocity prediction.  When a 256x256 image is patchified at 16x16, each RGB
token has 768 observed dimensions, equal to JiT-B's hidden width; direct
$x$-prediction is then dramatically better than predicting noise or velocity.
At 64x64 with patch 4, however, the token dimension is only 48 versus hidden
width 768, and the nine prediction/loss combinations are close.  Under
velocity loss, direct velocity prediction is slightly best in that reported
low-dimensional table (FID 3.46 versus 3.55 for direct $x$-prediction).

CAP-1R has only

\[
3\cdot2^2=12
\]

observed values per patch against hidden width 384.  It is even farther from
the high-dimensional-token failure regime.  Direct velocity prediction is
therefore retained.  Clean-image prediction becomes a high-priority change
only if future scaling uses aggressive patches for 256x256 or larger images.

### 18.6 Why full PixelFlow and EPG are deferred

PixelFlow is the most promising architectural follow-up because it shares
CAP-1R's encoder-free pixel-space flow premise.  Its target-64 ablation uses a
patch-2 token grid: a moderate 8x8 kickoff grid, corresponding to a 16x16
kickoff image, obtains FID 3.21 versus 3.34 for the non-cascaded 32x32 kickoff
grid.  An excessively small 2x2 kickoff grid worsens FID to 3.49 and lowers
recall.  The analogous target-32 proposal would be an 8 -> 16 -> 32 image
cascade, not a 1 -> 2 -> ... cascade.

That result is encouraging but insufficient to replace the first run:

- it is based on ImageNet-1K, an XL transformer, and 600k updates;
- the gain is small and some other metrics do not improve;
- training targets become stage-specific endpoint differences;
- inference must reproduce resolution transitions exactly;
- the correct interaction between a cascaded path, minibatch OT, and the
  formalization-derived B1/B2 trajectory has not been tested.

The official implementation is valuable as the authority for a later port,
but a simplified reimplementation would add more failure modes than it removes
before CAP-1R has established a working single-stage generator.

EPG answers a different question.  It removes the *external VAE* but explicitly
pretrains an internal encoder for 600k steps, attaches a random decoder, and
then fine-tunes the whole network.  Its smallest complete model is about 116M
parameters; the paper reports 57 hours on eight H200 GPUs for the Base encoder
pretraining alone.  This is not a mechanism that can be compressed honestly
into one day on 6 GB, and it would weaken the clean claim that CAP-1R uses no
separately trained representation model.  EPG is evidence that internal
representation learning can eventually scale pixel generation, not a local
recipe.

### 18.7 CAP-1R frozen run card

The exact revised foundation schedule is:

| Item | CAP-1R value |
|---|---:|
| Model | the Section 5 patch-2, width-384, 13-executed-block U-ViT |
| Path | the Section 3 linear noise-to-data path with exact multiscale minibatch assignment |
| Prediction | direct velocity |
| Updates 0--59,999 | $t=\sigma(u)$, $u\sim N(-0.8,1)$ |
| Updates 60,000--99,999 | $t\sim U[0,1]$ |
| Both 25k continuations | $t\sim U[0,1]$ |
| Loss | unweighted per-pixel velocity MSE; no importance correction |
| Optimizer | retain the Section 6.1 optimizer for the first run |
| Sampler | deterministic Heun at NFE 1/8/32, plus a fixed-latent NFE-64 secondary audit |
| Fork | unchanged paired control versus protected B1/B2 continuation |

The optimizer is deliberately not changed in the same revision.  Direct flow
references commonly use $\beta_1=0.9$ and little or no weight decay, whereas
official U-ViT CIFAR uses $(0.99,0.999)$ and weight decay 0.03.  There is no
controlled result identifying which side transfers better to a 5,000-image
U-ViT.  Changing optimizer statistics together with timestep sampling would
make a success or failure harder to interpret.  The current setting has the
stronger small-data/topology precedent and remains frozen.  A flow-optimizer
variant is a separate future experiment, not an invisible protocol edit.

Numerically, generate logit-normal times as `sigmoid(u)` in float32 and clamp
only to the representable open interval used by the model, for example
`[1e-5, 1-1e-5]`.  Uniform draws should receive the same clamp so implementation
details do not distinguish the phases.  The switch is keyed to the restored
global update counter, never to an epoch counter or wall time.

### 18.8 Additional mandatory preflight and logging

Add the following checks before the long run:

1. With a large fixed Monte Carlo sample, verify that the first-phase empirical
   `mean(logit(t))` is within 0.03 of -0.8 and its standard deviation is within
   0.03 of 1.0.
2. Verify algebraically in code that replacing published $s$ by `1 - s`
   produces exactly the CAP-1R $t$ stream to floating-point tolerance.
3. Verify that checkpoint/resume at updates 59,999, 60,000, and 60,001 restores
   the same next timestep and applies the switch exactly once.
4. At checkpoints 20k, 40k, 60k, 80k, and 100k, evaluate velocity MSE on a
   fixed training-only audit stream in 20 equal-width time bins.  These curves
   are diagnostics, not adaptive switch criteria.
5. Log the sampled-time histogram, per-bin sample counts, and per-bin audit
   MSE.  A scalar average loss is insufficient to detect endpoint neglect.
6. Keep the 100k recognizability gate unchanged.  The curriculum cannot waive
   collapse, memorization, or clipping checks.

No interim image metric may move the 60k switch.  The published switch point
is somewhat sensitive, and adapting it on a single local run would turn the
test into undocumented hyperparameter search.

### 18.9 Ranked follow-ups after CAP-1R

If CAP-1R's foundation is recognizable but training compute is clearly the
bottleneck, investigate in this order:

1. **PixelFlow-lite, one factor only:** replace the single path by a faithfully
   tested 8 -> 16 -> 32 cascade while keeping the model, optimizer, and
   evaluation fixed.  First prove stage endpoints and resolution transitions
   on analytic data; do not add B1/B2 until the cascade alone works.
2. **Flow optimizer ablation:** compare the frozen U-ViT optimizer with the
   direct-flow consensus $(\beta_1,\beta_2)=(0.9,0.999)$ and zero weight decay,
   using identical streams and more than one seed.
3. **SiD2-derived weighting:** test the explicitly derived $w_b^{(v)}(t)$ as a
   one-factor ablation with a predeclared, numerically capped weight and a
   small shift grid selected without test images.
4. **Prediction-space transition:** use direct $x$-prediction with velocity
   loss only when patch dimension becomes a genuine bottleneck at higher
   resolution; do not infer a 32x32 gain from JiT.
5. **EPG-scale representation pretraining:** only after multi-GPU compute is
   available and only if the research question permits an internal separately
   pretrained encoder.

Very recent proposals such as spectral residual corrections can be useful
ablation ideas, but they are not mature enough to displace the better-vetted
mechanisms above.  CAP-1 already has an explicitly audited spectral term in
B1; adding another learned spectral module would also blur whether the gain
came from the formalized discrepancy or the backbone.

### 18.10 Final assessment

The two user-supplied sources are valuable, but neither by itself supplies the
local mechanism:

- SiD2 validates encoder-free pixel generation and supports patch-2 U-ViT
  compute allocation, while its best loss belongs to a different diffusion
  parameterization and resolution regime.
- EPG demonstrates that a model can learn its own internal representation
  instead of inheriting a VAE, but its separate pretraining phase is far beyond
  the budget and is not strictly representation-pretraining-free.

The most reliable actionable addition comes from the broader search:
Curriculum Sampling matches pixel resolution, dataset, linear CFM objective,
and 100k target budget unusually closely.  Its time-coordinate mapping is
exact, it adds essentially no compute, and the final uniform phase limits the
cost of imperfect transfer.  CAP-1R therefore keeps the proven pieces simple
and spends the scarce run on the single new mechanism with the best ratio of
direct evidence to implementation risk.
