# Local S3 Pixel MeanFlow Pre-Run Audit

**Status:** all launch gates passed on matching frozen hashes. The authorized
two-unit run began in the background on 2026-08-02 at 22:28 local time.

## Scope and claim

S3 asks whether a resource-scaled raw-pixel MeanFlow can form a recognizable,
diverse, exactly one-network-call generator for the single CIFAR-10 automobile
class. It is a local capability gate, not a reproduction of ImageNet pMF, not
yet a drifting correction, and not evidence of superiority to the paper.

## Mathematical and literature audit

The implementation follows the pMF denoised-image parameterization

\[
  u_\theta(z_t,r,t)=\frac{z_t-x_\theta(z_t,r,t)}{\max(t,0.05)}
\]

and the iMF compound field

\[
  V_\theta=u_\theta+(t-r)\operatorname{stopgrad}
  \operatorname{JVP}[u_\theta;(v_\theta,1,0)].
\]

For this unconditional model, `v_theta = u_theta(z,t,t)` is not an ad-hoc
shortcut. Improved MeanFlow explicitly gives this boundary construction as
the parameter-free predicted marginal-velocity tangent and reports that it
addresses the conditional-tangent variance problem. Its auxiliary velocity
head improves large-scale results but adds training capacity; it is not
required for the validity of the objective. The pMF paper's own Algorithm 1
uses this boundary construction.

The implementation uses the complete time triangle, with exactly 50% of each
batch on `r=t`; the diagonal value is the first logit-normal draw, assigned
before sorting. The JVP argument order and tangent are `(z,t,r)` and
`(v_theta,1,0)`. Only the JVP result is stopped, leaving the ordinary gradient
through `u_theta`. Direct image prediction makes the endpoint at `(r,t)=(0,1)`
exactly one model call.

One mismatch found during audit was repaired: when `t<0.05`, the released pMF
implementation uses a clamp-consistent target `(z_t-x)/max(t,0.05)`, whereas
the first draft always used `epsilon-x`. The two agree outside this extremely
small time region. A regression now covers the clamped case.

## Architecture and optimization decisions

- The model consumes raw RGB pixels and predicts raw RGB pixels. There is no
  VAE, learned tokenizer, teacher, feature encoder, reference bank, or solver.
- A 12-block, width-384 U-shaped transformer with 4-by-4 patches has 64 image
  tokens and about 30.7M parameters. Long skips store encoder-block inputs;
  this avoids the erroneous deepest-state self-concatenation found and fixed
  during preflight.
- Four-pixel patches are a measured local-compute compromise. They preserve
  exact per-patch RGB prediction, but provide a coarser token grid than pMF's
  published 16-by-16 token grid. S3 therefore tests the local profile, not the
  maximum-quality pMF architecture.
- The initial audit sanity exposed rising JVP magnitudes and failed its fixed
  raw-velocity diagnostic. Inspection against the released backbone found two
  fidelity errors: residual branches were not zero-gated and body weights had
  variance `1/fan_in` instead of `0.1/fan_in`. Both are repaired, with
  regressions. A second sealed diagnostic then localized the problem: the
  flow-matching diagonal improved from `0.858` to `0.303`, while the interior
  correction worsened from `0.637` to `3.922`. This is precisely the tangent
  instability iMF's auxiliary marginal-velocity head addresses. The local
  model now includes a zero-initialized, training-only auxiliary pixel head
  sharing the backbone, with the published adaptive FM loss. It is used for
  the JVP tangent and excluded from inference. This sequence is why the full
  run remained interlocked rather than being launched after mechanical tests.
- The auxiliary diagnostic learned (`~0.27--0.33` raw error) but did not by
  itself remove the interior JVP spikes. A second architecture comparison
  found that the draft embedded absolute `t`, while released iMF/pMF expressly
  conditions only on `h=t-r`. The high-frequency `t` token created an
  unnecessary direct time derivative inside the JVP. It is removed; `t`
  remains only in the exact pixel/velocity conversion.
- AdamW is retained at the documented iMF constant rate `1e-4`. The pMF paper
  reports materially better large-scale FID
  with Muon, but Muon requires correct mixed parameter routing and its own
  tuned learning rate. Introducing it immediately before this run without a
  local controlled validation would exchange a known optimizer for a new
  confound. This is a conservative validity decision, not a claim that AdamW
  is optimal.
- Perceptual loss is deliberately excluded. Published pMF gains strongly from
  VGG/ConvNeXt features, but that would violate this experiment's strict
  encoder-independent mechanism. The likely cost is lower perceptual quality.
- With one fixed class, labels and classifier-free guidance are unnecessary.

## Reproducibility, leakage, and recovery repairs

The audit expanded the sealed source manifest to every transitive local module
used by training, metrics, provenance, grids, and launching. The authorization
digest is frozen at launch and every checkpoint/final shard rechecks it; an
edit during training aborts rather than producing mixed-source artifacts.

All data-order, augmentation, noise, triangle-value, and diagonal-mask streams
are independent and restartable. Checkpoints preserve the model, EMA,
optimizer, all stochastic states, history, source/profile identities, and
cumulative elapsed time. Exact interrupted-versus-uninterrupted replay is a
regression test. A resumed runner validates and skips completed shards instead
of refusing them, while incompatible or corrupt shards/checkpoints fail hard.

Training for both fixed units is completed before any test feature, test-based
memorization scale, or grid is computed. Evaluation then uses one shared,
sealed Gaussian prior across units, making the replication comparison paired
rather than confounded by different evaluation noise. Test metrics are
report-only and cannot select training checkpoints or alter the run.

## Evaluation limits

The report includes FID, KID, improved precision/recall, effective rank,
duplicate/diversity statistics, spectrum slope, range checks, and nearest
training-image diagnostics. At 1,000 examples, absolute FID is biased and is
not comparable to published 50,000-sample ImageNet FID. KID is unbiased but
still noisy; precision/recall and the fixed uncurated grids must be read
together. Both units must be noncollapsed and nonmemorizing. No single scalar
can override a visibly failed one-step generator.

## Remaining launch gates

The launch authorization may be created only after all of these are green:

1. formatting/static checks and all encoder-independent regression tests;
2. a 1,000-update, train-split-only learning sanity showing reduced fixed
   diagnostic raw velocity error, finite changed parameters, and nonconstant
   EMA endpoints without inspecting images or test features;
3. a fresh full-shape CUDA preflight after the final source edit;
4. source/profile/artifact hash verification;
5. a GPU-process and disk-space check showing the device is free for the
   sequential two-unit run.

## Learning-sanity result

After the fidelity repairs above, the sealed 1,000-update CUDA sanity passed.
On fixed train-only diagnostic batches, overall raw velocity MSE fell from
`0.7471` to `0.4182`; the diagonal fell from `0.8576` to `0.2845`, and the
interior fell from `0.6367` to `0.5519`. The training-only auxiliary raw MSE
was stable near `0.25--0.32` late in the run. EMA endpoints were finite,
nonconstant, and had effective rank `38.6` across 64 samples. No test image,
Inception feature, sample grid, or human visual judgment was used.

Artifact:
`numerics/encoder_independent_drifting/stage_pmf/s3_learning_sanity.json`
with sidecar SHA-256
`bbb856297834dde83116ab866d9173789bfadbbc9bd7e2e7766814e1acc1d5db`.

## Final mechanics and cost result

The final full-shape CUDA preflight passed on the RTX 4050 Laptop GPU. It
measured 30,499,296 training parameters (30,480,816 inference parameters),
3,722,169,344 peak allocated bytes, and 0.2369 seconds per update over the
short warmed check. The projected sequential time for both 60,000-update units
is 7.90 hours. The constant-field identity error was below `4e-8`; the
float64 central finite-difference/JVP relative error was `1.34e-5`; the one-step
sampler made exactly one model call; and official class counts were
5,000 train / 1,000 test.

Artifact:
`numerics/encoder_independent_drifting/stage_pmf/s3_preflight_fullshape.json`
with sidecar SHA-256
`42d46156b3a22e3f70f7a0c47f700d31033adb0c82ec38cfdb39edd96055fdf6`.

Both final artifacts bind the identical source digest
`df056d0f2edf03536f30f7ef622055da451c31aeffc9f71f82cb11875e682a77`
and profile digest
`10620652e0eb2f96b4bf29a196c9a6259888dea5886e745d5690d487e9292ff8`.

## Primary references

- Geng et al., [Improved Mean Flows: On the Challenges of Fastforward
  Generative Models](https://arxiv.org/abs/2512.02012), especially the
  predicted-marginal-velocity tangent and boundary construction.
- Lu et al., [One-step Latent-free Image Generation with Pixel Mean
  Flows](https://arxiv.org/abs/2601.22158v3), especially Algorithm 1, the
  denoised-image conversion, full-triangle ablation, and Appendix A.
- [Official pMF implementation](https://github.com/Lyy-iiis/pMF), used to
  cross-check the time sampler, denominator floor, adaptive loss, and released
  auxiliary-head variant.
