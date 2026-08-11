# Local S3 Pixel MeanFlow Foundation Protocol

**Status:** implementation complete through the audit boundary; **the full
two-unit run is not authorized**. The runner requires a separate, hash-bound
post-audit authorization file that this build deliberately does not create.

## Scientific question

Can a raw-pixel, latent-free model learn a recognizable and diverse
one-network-call generator for one complete CIFAR-10 class before any
Sinkhorn/drifting correction is attached?

This is a capability gate. A failure means the one-step foundation, not the
distributional correction, is the current bottleneck. A success permits the
later matched S4 fork; it does not by itself establish an improvement over the
paper or a pure drifting result.

## Target and leakage boundary

- Target: CIFAR-10 `automobile` (official label 1).
- Training: all 5,000 automobiles in the official training split.
- Reporting: all 1,000 automobiles in the official test split.
- Labels select the one-class task once; labels are never an input to the
  network, objective, sampler, or training controller.
- Horizontal flip is the only augmentation.
- No test image, test statistic, sample grid, FID, KID, or learned feature may
  affect training, checkpoint choice, or launch configuration.

## Model and one-step identity

The network is a 32-by-32 raw-pixel U-shaped Vision Transformer with 4-by-4
patches, long token skips, and the scalar condition `h=t-r`. The call signature
also carries `t`, but—following released iMF/pMF—absolute time is used only by
the analytic pixel/velocity conversion and is not embedded as a network token.
Its direct
output is a denoised-image prediction

\[
  \hat x_\theta(z_t,t,h),
\]

not a latent and not a velocity. Average velocity is derived as

\[
  u_\theta(z_t,r,t)
  =\frac{z_t-\hat x_\theta(z_t,t,t-r)}{\max(t,0.05)}.
\]

At inference, `(r,t)=(0,1)`, hence

\[
  z_0=z_1-u_\theta(z_1,0,1)=\hat x_\theta(z_1,1,1).
\]

The exported sampler therefore contains exactly one model invocation. There
is no solver, VAE, reference bank, kernel, Sinkhorn solve, CFG second pass, or
training JVP at inference.

## Training identity

Draw clean pixels `x`, independent Gaussian noise `e`, and

\[
z_t=(1-t)x+t e,\qquad v=e-x.
\]

Draw two times independently from `sigmoid(N(0.8,0.8^2))`; for exactly half
of each microbatch replace the second draw by the first, then sort into
`r <= t`. This gives `r=t` on exactly half the batch without biasing those
diagonal times toward the maximum of two draws. It covers both
the flow-matching diagonal and the interior of the complete MeanFlow triangle.

For this unconditional task, a training-only auxiliary head sharing the
transformer backbone predicts the marginal boundary velocity used by the JVP.
It has its own adaptive flow-matching loss. This is the resource-scaled form
of improved MeanFlow's auxiliary-head option; the head is never evaluated at
inference. The parameter-free identity remains the fallback/check:

\[
v_\theta(z_t,t)=u_\theta(z_t,t,t).
\]

With arguments ordered `(z,t,r)`, compute

\[
J_\theta=\operatorname{JVP}
 [u_\theta;(v_\theta,1,0)],\qquad
V_\theta=u_\theta+(t-r)\operatorname{stopgrad}(J_\theta).
\]

The main and auxiliary losses are pMF's adaptively normalized velocity
regressions:

\[
\tilde v_i=\frac{z_{t,i}-x_i}{\max(t_i,0.05)},\qquad
\ell_i=\|V_{\theta,i}-\tilde v_i\|_2^2,
\qquad
\mathcal L=\frac1B\sum_i
\frac{\ell_i}{\operatorname{stopgrad}(\ell_i+0.01)}.
\]

The auxiliary term has the same form with its predicted instantaneous
velocity. The two normalized terms are added, following iMF/pMF.

Except in the extremely small sampled region `t < 0.05`, `v-tilde=e-x`; the
clamped form keeps the target consistent with the stabilized conversion used
by the released pMF implementation. The JVP contribution is stopped, so
training is first-order. The ordinary
gradient through `u_theta` remains. The boundary tangent is also detached
because it can influence only the already-stopped JVP contribution.

## Candidate local profile (audit pending)

| item | value |
|---|---:|
| backbone | raw-pixel U-ViT + training-only auxiliary velocity head |
| patch / token count | 4 / 64 image tokens |
| width / depth / heads | 384 / 12 / 8 |
| prediction | direct RGB pixels |
| optimizer | AdamW, betas `(0.9,0.95)`, no weight decay |
| learning rate | `1e-4` |
| microbatch / accumulation | 16 / 1 (effective 16) |
| updates | 60,000 per unit (192 nominal class epochs) |
| EMA | 0.9999 |
| gradient clipping | 1.0 |
| checkpoints | 2k, 10k, 30k, 60k |
| initial units | 700 and 701 |

These values are implemented so they can be timed and audited; they are not
treated as immutable until the preflight and review accept them. The initial
two units are fixed independently of outcomes. There is no optional third run
inside this protocol.

The first preflight candidate used 2-by-2 patches, an effective batch of 64,
and 30,000 updates. A measured RTX-4050 full-shape update projected to about
325 sequential hours for two units, so that candidate was rejected before any
quality result existed. Outcome-blind shape timing showed that 4-by-4 patches
retain direct pixel prediction while removing the quadratic 256-token
attention bottleneck. The current 30.6M-parameter, batch-16 candidate used
about 3.37 GiB and 0.236 seconds per update over a short warmed measurement;
60,000 updates project to roughly 3.9 hours per unit, or 7.9 hours sequential.
The formal preflight artifact, rather than this short estimate, is the value
the audit must accept or reject.

The transformer follows the released stabilization at initialization: linear
weights have variance `0.1/fan_in`, and each attention/MLP residual branch has
a learned scale initialized at zero. AdamW is used deliberately at this stage:
the official pMF work reports Muon
as faster and better, but the local repository has no independently audited
Muon implementation. Substituting an unverified optimizer immediately before
the foundation run would create a larger correctness risk than the expected
speed benefit. The constant `1e-4` rate matches the documented Adam iMF
profile. A separately tested Muon amendment can be proposed before
authorization if the audit supports it.

## Preflight gates

Before launch, all of the following must pass:

1. direct pixel-output shape and deterministic initialization;
2. exact triangle support and exact 50% diagonal allocation;
3. constant-velocity analytic JVP;
4. central finite-difference agreement for a nontrivial tiny network;
5. derivative stop-gradient placement;
6. `(x,u)` conversion round trip;
7. exactly one model call in endpoint sampling;
8. deterministic replay of stochastic streams and tiny training;
9. finite loss, parameters, and gradients;
10. official split counts of 5,000/1,000 and immutable split digests;
11. measured full-profile GPU memory and seconds per update;
12. a source/config manifest sealed before authorization.

The preflight is outcome-blind: it may inspect mechanics, loss finiteness,
runtime, and memory, but not sample quality or learned metrics.

## Full-run artifacts and later gate

Each unit will preserve source-checked restartable checkpoints containing the
model, EMA, optimizer, data cursor, and every stochastic generator state. Only
after both fixed units complete will the runner compute report-only FID, KID,
precision/recall, effective rank, duplicate/diversity/range diagnostics,
nearest-training-image memorization statistics, uncurated fixed-noise grids,
wall time, peak memory, parameter count, and an inference-call audit.

The test split and Inception representation are evaluation instruments only.
They do not make the training procedure encoder-dependent.

S3 advances only if the fixed endpoint yields recognizable, nonconstant,
diverse automobile images without a memorization veto in both units. Numerical
metric improvement alone cannot override visibly failed generation, and a
visually appealing grid cannot override collapse or copying.

## Launch interlock

`run_two_unit.py` refuses a non-dry run unless an external
`s3_launch_authorization.json` contains:

- status `AUTHORIZED-AFTER-S3-AUDIT`;
- the exact profile name and profile digest;
- the exact source-manifest digest;
- units `[700,701]`.

This file is intentionally absent. Source or protocol edits after an audit
invalidate any older authorization automatically.

## Primary references

- Geng et al., [Mean Flows for One-step Generative
  Modeling](https://arxiv.org/abs/2505.13447).
- Lu et al., [One-step Latent-free Image Generation with Pixel Mean
  Flows](https://arxiv.org/abs/2601.22158), especially the denoised-image field,
  full-triangle ablation, and stopped-JVP objective.
- [Official pixel MeanFlow implementation](https://github.com/Lyy-iiis/pMF),
  cross-checked for the denominator floor, time sampler, adaptive weighting,
  and one-step update.

The local model is a resource-scaled unconditional implementation of those
identities, not a reproduction of the published ImageNet system or its FID.

## Audit commands (full launch intentionally omitted)

From the repository root:

```powershell
# S3 tests plus all existing encoder-independent regressions (CPU wheels)
uv run --python 3.12 --with torch==2.7.1 `
  --with torchvision==0.22.1 --with numpy --with scipy --with pillow `
  python -m numerics.encoder_independent_drifting.tests.run_all

# Full-shape, five-update mechanics/memory/timing preflight on CUDA
uv run --python 3.12 `
  --extra-index-url https://download.pytorch.org/whl/cu126 `
  --index-strategy unsafe-best-match `
  --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 `
  --with numpy --with scipy --with pillow `
  python -m numerics.encoder_independent_drifting.stage_pmf.preflight `
  --profile local_s3 --updates 5 --device cuda `
  --out numerics/encoder_independent_drifting/stage_pmf/s3_preflight_fullshape.json

# Prove the frozen-shaped runner resolves without permitting training
uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy `
  --with pillow `
  python -m numerics.encoder_independent_drifting.stage_pmf.run_two_unit `
  --dry-run
```

There is deliberately no executable full-launch command in this build. The
post-audit authorization and launch invocation belong to the next decision,
not this implementation pass.
