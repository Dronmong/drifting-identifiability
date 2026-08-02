# MeanFlow-Drifting: one-step, encoder-free image generation

## Status and purpose

This is the live research document for combining pixel MeanFlow with an
encoder-independent distributional drifting mechanism. Sections 1--13 record
the initial Laplace-field proposal. Sections 14 onward contain the deeper
literature audit and the revised recommendation: retain the Laplace field as a
formalization-derived diagnostic, but use a fully balanced Sinkhorn
Wasserstein-gradient-flow target as the primary distribution correction.

The immediate objective is a focused but credible image experiment:

> Train a raw-pixel, VAE-free generator on one complete CIFAR-10 class, retain
> exactly one neural-network evaluation at inference, and test whether an
> encoder-free Laplace drift objective improves a matched one-step generator.

The proposal deliberately separates three claims:

1. **Pixel MeanFlow capability:** a direct pixel generator can produce useful
   samples in one network call without a VAE or training encoder.
2. **Drift attribution:** adding the formalized Laplace field improves a
   matched MeanFlow continuation.
3. **Pure drifting:** the drifting objective alone can maintain or improve a
   competent generator after initialization.

Passing (1) and (2) would establish an encoder-free, drift-regularized,
one-step generator. It would not establish that pure drifting trains an image
generator from noise. Claim (3) requires its own takeover experiment.

## 1. Why MeanFlow is the right one-step foundation

Ordinary flow matching learns an instantaneous velocity and normally evaluates
that velocity repeatedly at inference. MeanFlow instead learns an **average
velocity over an interval**, so a complete noise-to-data interval can be
traversed in one network evaluation.

Let (x\sim p_{\mathrm{data}}), let \(\epsilon\sim\mathcal N(0,I)\), and use
the linear conditional path

\[
z_t=(1-t)x+t\epsilon,\qquad v=\epsilon-x.
\]

The population average velocity from time (t) back to (r\le t) is

\[
u(z_t,r,t)=\frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,d\tau.
\]

MeanFlow's defining identity is

\[
u=v-(t-r)\frac{d}{dt}u,
\]

where the total derivative follows the trajectory:

\[
\frac{d}{dt}u
=\partial_z u\,v+\partial_tu.
\]

Equivalently,

\[
u+(t-r)\frac{d}{dt}u=v.
\]

This identity is what turns interval integration into a regression problem
that can be trained with a Jacobian-vector product (JVP).

## 2. Use pixel MeanFlow, not the original velocity parameterization

The network should directly predict an image-like quantity

\[
\widehat x_\theta(z_t,r,t)=\operatorname{net}_\theta(z_t,t,t-r).
\]

Convert this prediction into an average velocity:

\[
u_\theta(z_t,r,t)
=\frac{z_t-\widehat x_\theta(z_t,r,t)}{\max(t,0.05)}.
\]

The denominator clamp matches the released pMF implementation and prevents
instability near (t=0). At inference (t=1), so the clamp does not alter the
one-step formula.

Direct (x)-prediction is not a cosmetic choice. Pixel MeanFlow reports that
at ImageNet 256 resolution, under the same MSE/Muon ablation, (x)-prediction
achieves FID 9.56 while (u)-prediction collapses to FID 164.89. The proposed
explanation is that the image target lies near a low-dimensional data manifold,
whereas a noisy velocity has full-dimensional support.

## 3. Improved MeanFlow training identity

For the unconditional one-class experiment, use the boundary representation
of the instantaneous velocity:

\[
v_\theta(z_t,t)=u_\theta(z_t,t,t).
\]

Then compute the directional derivative

\[
J_\theta
=\operatorname{JVP}
  \left[u_\theta;(v_\theta,0,1)\right],
\]

where the function arguments are ordered as ((z,r,t)). The predicted
compound velocity is

\[
V_\theta
=u_\theta+(t-r)\operatorname{stopgrad}(J_\theta).
\]

The primary loss is

\[
\mathcal L_{\mathrm{pMF}}
=\mathbb E\left[\|V_\theta-(\epsilon-x)\|^2\right].
\]

The predicted boundary velocity is used as the JVP tangent; the regression
target remains the conditional velocity (epsilon-x). The JVP contribution is
stop-gradient, as in the official implementations. This avoids second-order
optimization while retaining gradients through the directly evaluated
(u_\theta).

Use pMF's adaptive per-sample weighting. If

\[
\ell_i=\|V_{\theta,i}-(\epsilon_i-x_i)\|^2,
\]

then

\[
\mathcal L_{\mathrm{adaptive}}
=\frac1B\sum_i
\frac{\ell_i}
     {\operatorname{stopgrad}((\ell_i+0.01)^p)},
\qquad p=1.
\]

The boundary variant is preferred initially because the experiment is
unconditional. Improved MeanFlow found it stronger than the auxiliary-head
variant without CFG, while requiring no training-only branch.

## 4. Exactly one network evaluation at inference

At inference, sample (z_1\sim\mathcal N(0,I)) and evaluate the endpoint
((r,t)=(0,1)):

\[
u_\theta(z_1,0,1)=z_1-\widehat x_\theta(z_1,0,1).
\]

The one-step MeanFlow update therefore becomes

\[
z_0=z_1-u_\theta(z_1,0,1)
=\widehat x_\theta(z_1,0,1).
\]

The deployed sampler is simply:

```python
noise = torch.randn(batch, 3, 32, 32)
images = model(noise, t=1.0, h=1.0)
```

There is no ODE solver, iterative correction, VAE, kernel evaluation, reference
bank, CFG second pass, or auxiliary head at inference.

## 5. The full time triangle is mandatory

Training must cover

\[
0\le r\le t\le1.
\]

Pixel MeanFlow reports that training only on (r=t), only on (r=0), or on
the union of those boundary lines all fails. The interior of the triangle is
therefore not optional.

The initial implementation should follow the released pMF sampler:

- sample (r,t) independently from a logit-normal law;
- use `P_mean = 0.8`, `P_std = 0.8`;
- sort the pair so (r\le t);
- set (r=t) for 50% of each batch;
- condition the network on (t) and (h=t-r);
- retain the other 50% as genuine interval samples.

The one-class experiment should not use CFG. Removing labels, unconditional
branches, guidance intervals, and extra guidance evaluations makes the claim
cleaner and substantially reduces training cost.

## 6. The initial formalization-derived drift regularizer

MeanFlow supplies a capable one-step map. The repository's contribution is a
population-correctness signal derived from the normalized Laplace field.

For probes (a_i), independent real positives (x_j), and one-step generated
negatives (y_k\), define

\[
\widehat V_\tau(a_i)
=\sum_j
  \frac{e^{-\|a_i-x_j\|/\tau}}
       {\sum_\ell e^{-\|a_i-x_\ell\|/\tau}}x_j
-\sum_k
  \frac{e^{-\|a_i-y_k\|/\tau}}
       {\sum_\ell e^{-\|a_i-y_\ell\|/\tau}}y_k.
\]

The empirical zero-set energy is

\[
\mathcal L_{\mathrm{drift}}
=\frac1M\sum_i\|\widehat V_\tau(a_i)\|^2.
\]

This is the B2 mechanism already tested in the repository. It is not the F1
procedure that repeatedly applies the raw drift map. F1 collapsed diverse
starts into a low-rank attractor. B2 instead treats zero field as a
distributional regularizer and reduced paired held-out raw drift energy by
approximately 23--27% in its successful units.

The roles must remain sample-split:

- **probes:** real samples plus small continuous Gaussian noise;
- **positives:** an independent augmented real batch;
- **negatives:** fresh one-step endpoint samples;
- **gradient path:** only through the generated negatives;
- **detached:** probes and positives.

The initially proposed hybrid objective is

\[
\mathcal L_{\mathrm{hybrid}}
=\mathcal L_{\mathrm{pMF}}
+\lambda_{\mathrm{drift}}\mathcal L_{\mathrm{drift}}.
\]

At the population level, the objectives are compatible: when the generated
law equals the data law, the normalized Laplace field vanishes. This does not
prove that stochastic joint optimization converges, nor does it make direct
field-energy minimization a safe particle objective. The deeper audit in
Sections 14--22 therefore demotes this loss to a diagnostic and ablation.

## 7. Fixed multiscale geometry without a learned encoder

The raw-pixel Laplace term is theorem-aligned but may not capture coarse image
geometry efficiently. Add only deterministic, auditable views:

\[
T_0(x)=x,\qquad
T_1(x)=\operatorname{avgpool}_2(x),\qquad
T_2(x)=\operatorname{avgpool}_4(x).
\]

For each view, compute its own field and nonnegative energy (E_s), then use

\[
\mathcal L_{\mathrm{multi}}
=\alpha_0E_0+\alpha_1E_1+\alpha_2E_2,
\qquad \alpha_s>0.
\]

The losses must be computed separately and summed; fields from different
views must not be averaged before squaring, because cancellation could hide a
nonzero field. Keeping the identity view (T_0) with positive weight preserves
the population identifiability implication: zero total energy forces
(E_0=0), and the raw-space converse can then apply.

Each view receives an independently calibrated and frozen bandwidth. Use the
existing outcome-blind B2 rule: choose \(\tau_s\) so the off-diagonal kernel
effective-sample-size fraction is approximately 0.60 on training data. Report
the lower ESS quantiles and maximum-weight quantiles, not only the median.

The first confirmatory comparison should retain raw B2 as the primary drift
arm. Multiscale drift is a development arm until its interaction with pMF has
been tested.

## 8. Architecture and optimization

The proposed backbone is a roughly 45M-parameter pixel U-ViT-S/2:

- 32 by 32 RGB input;
- patch size 2;
- 256 image tokens;
- width 512;
- depth 12;
- eight attention heads;
- long U-shaped residual connections;
- direct RGB (x)-prediction output;
- time conditioning on (t) and (h=t-r).

This is large enough to be a serious image generator but smaller than the
89--131M published MeanFlow/pMF base models, which are poorly matched to a
5,000-image target and a 25-dollar budget.

Use Muon if and only if its PyTorch implementation passes a deterministic
optimizer preflight. The pMF reference uses:

- learning rate (10^{-3});
- Adam second-moment coefficient 0.95 inside Muon;
- no weight decay;
- EMA for evaluation.

No LPIPS, VGG, ConvNeXt, VAE, pretrained teacher, learned OT metric, or learned
feature loss may affect training, bandwidths, checkpoint activation, or model
selection. Learned networks may be used for final report-only FID/KID/precision
and recall, but the evaluated checkpoint must be selected by a frozen step or
nonlearned training criterion.

## 9. Protected drift weighting

Reuse the validated B2 calibration rather than guessing an absolute
coefficient. On a calibration batch, compute parameter-gradient norms

\[
g_M=\nabla_\theta\mathcal L_{\mathrm{pMF}},\qquad
g_D=\nabla_\theta\mathcal L_{\mathrm{drift}}.
\]

For a drift event, set

\[
\lambda_{\mathrm{event}}
=0.25\frac{\|g_M\|_2}{\|g_D\|_2}.
\]

Apply the drift event every tenth update. The event-level ratio is 25%, while
the cadence-averaged nominal gradient budget is approximately 2.5%. Freeze
the resulting coefficient before the matched comparison.

Log at every drift event:

- unweighted and weighted gradient norms;
- cosine similarity between (g_M) and (g_D);
- raw and multiscale drift energies;
- ESS and maximum-weight quantiles;
- generated effective rank and clipping fraction;
- wall time and peak memory.

Do not silently introduce PCGrad in the primary arm. A projected-gradient arm
may be preregistered as a rescue experiment if persistent negative gradient
cosines are observed in development, but it changes the optimization mechanism
and must be reported separately.

## 10. Implementation preflight

Before cloud training, require all of the following:

1. A constant-velocity analytic case gives (u=v) and zero JVP correction.
2. At (r=t), the multiplier (t-r) removes the derivative correction.
3. (x\leftrightarrow u) conversion round-trips numerically.
4. `torch.func.jvp` agrees with a central finite-difference directional
   derivative on a tiny deterministic model.
5. The JVP tangent and derivative term are detached exactly as specified.
6. Single-device and two-device gradients agree on a tiny test; PyTorch JVP
   can otherwise bypass DDP synchronization silently.
7. The inference forward counter is exactly one.
8. Removing the training-only JVP machinery leaves identical inference output.
9. Drift probes and positives have no gradients; generated negatives do.
10. Drift energy is nonnegative and uses mean squared field norm, never the
    squared norm of the mean field.
11. Bandwidth calibration reproduces its frozen ESS target.
12. A 1,000--2,000-update sanity run lowers the pMF loss, remains finite, and
    produces nonconstant endpoint images.

## 11. Staged experiment

### Foundation

Train on the complete 5,000-image CIFAR-10 automobile training class. Seal the
1,000 automobile test images until final reporting.

Use:

- raw pixels normalized consistently to the noise scale;
- independent Gaussian noise, not minibatch OT, in the primary pMF arm;
- horizontal flips only;
- complete ((r,t))-triangle sampling;
- direct (x)-prediction;
- unconditional boundary-velocity iMF loss;
- EMA checkpoints preserved at fixed intervals.

The exact number of updates must be set after benchmarking the complete JVP
training step. The run must fit one day and leave budget for a matched fork.
The earlier CAP target of 160,000 batch-64 updates corresponds to 2,048 nominal
presentations of the 5,000-image class, but it is a ceiling, not a value to
launch without a measured seconds-per-step projection.

### Matched continuation

Clone model parameters, EMA, optimizer, data cursor, and RNG states:

| Arm | Continuation |
|---|---|
| M0 | pMF only |
| M1 | pMF + raw identity-space B2 drift |
| M2 development | pMF + raw and fixed multiscale drift |

M1 versus M0 is the primary causal comparison. M2 is a development question
about whether deterministic coarse geometry improves the raw field.

### Drift-only takeover diagnostic

From the same successful foundation, optionally run a short arm with

\[
\mathcal L=\mathcal L_{\mathrm{drift}}
\]

and pMF loss disabled. Success would support the stronger statement that drift
can continue improving a competent one-step generator. Failure would not
invalidate the hybrid result; it would show that MeanFlow remains an active
stabilizer.

## 12. Evaluation and claim gates

Primary reporting should include:

- uncurated sample grids at fixed checkpoints and fixed noise seeds;
- KID and FID against the sealed one-class test set, with uncertainty;
- precision and recall;
- effective rank and occupancy/coverage diagnostics;
- nearest-neighbor memorization checks against the training set;
- clipping/saturation rate;
- one-NFE forward-counter audit;
- drift-energy reduction on fresh sample-split batches;
- wall time, memory, parameters, and inference latency.

Promotion requires M1 to improve image quality or coverage over M0 while
passing diversity and memorization vetoes. Lower drift energy alone is only a
mechanism result.

If M1 wins, the correct claim is:

> A one-step, VAE-free, externally encoder-free pixel generator whose matched
> performance is improved by a formalization-derived Laplace drifting
> objective.

Do not claim that MeanFlow itself is a Drifting Model, that pure drifting
trains from noise, or that the result already transfers to ImageNet.

## 13. Initial assessment (superseded by the deep audit below)

This route is substantially safer than the previous CAP formulation:

- the deployed generator is exactly one-step rather than a 50-NFE flow;
- the base capability is supported by MeanFlow and pixel MeanFlow results;
- direct image prediction avoids a documented high-dimensional target failure;
- full-triangle sampling avoids a documented endpoint-only failure;
- no external encoder, VAE, or perceptual network is used in training;
- the drift component reuses the repository's only positive image-space drift
  mechanism rather than the collapsed iterative drift map;
- a matched fork isolates whether drifting provides value beyond MeanFlow.

Confidence is high that a correctly implemented foundation can produce
recognizable one-class images. Whether the drift arm beats the matched pMF
control is the genuinely new and uncertain research question.

## Sources

### Primary papers and official implementations

1. Geng et al., **Mean Flows for One-step Generative Modeling**.
   <https://arxiv.org/abs/2505.13447>
2. Geng et al., **Improved Mean Flows: On the Challenges of Fastforward
   Generative Models**.
   <https://arxiv.org/abs/2512.02012>
3. Lu et al., **One-step Latent-free Image Generation with Pixel Mean Flows**.
   <https://arxiv.org/abs/2601.22158>
4. Official pixel MeanFlow JAX implementation.
   <https://github.com/Lyy-iiis/pMF>
5. Official PyTorch/GPU MeanFlow implementation and JVP/DDP notes.
   <https://github.com/Gsunshine/py-meanflow>
6. Lee and Chun, **Generative Modeling via Drifting**.
   <https://arxiv.org/abs/2602.04770>
7. Hoogeboom et al., **Simpler Diffusion (SiD2): 1.5 FID on ImageNet512 with
   pixel-space diffusion**, including the residual U-ViT design.
   <https://arxiv.org/abs/2410.19324>
8. **There is No VAE: End-to-End Pixel-Space Generative Modeling via
   Self-Supervised Pre-training**. This is useful as evidence about pixel-space
   representation learning, but its learned pretraining encoder is not part of
   the strict encoder-free proposal.
   <https://arxiv.org/abs/2510.12586>

### Repository evidence

9. `numerics/EncoderIndependentF1Results.md` and
   `numerics/EncoderIndependentF1K200ConfirmationResults.md`: repeated raw
   drift-map collapse.
10. `numerics/EncoderIndependentB0B1B2Results.md`: successful encoder-free
    bridge, spectral anchor, and B2 drift-energy mechanism tests.
11. `numerics/EncoderIndependentB2Analysis.md`: B2 methodology, ESS bandwidth,
    gradient calibration, tradeoffs, and corrected claim scope.
12. `numerics/EncoderIndependentB3Results.md`: matched-capacity failure of the
    raw one-step drifting proxy.
13. `numerics/CAPMax25ResearchPlan.md`: one-class target, architecture, budget,
    and evaluation principles; its multi-step deployed flow is superseded by
    the one-step pMF foundation proposed here.
14. `DriftingIdentifiability/LaplaceRnRoadmap.md` and the promoted Laplace
    converse modules: population identifiability for the raw Euclidean
    Laplace mean-shift field.

## Questions carried into the deep audit

The next pass must examine, rather than assume:

- whether the pMF regression and drift-energy gradients are aligned near and
  far from the data manifold;
- whether probes near real data sufficiently constrain missing generated
  modes;
- whether fixed multiscale views improve geometry without recreating a hidden
  feature encoder;
- whether pMF time sampling should remain fixed during drift continuation;
- whether the B2 bandwidth should depend on the target only or on a frozen
  target/generated mixture;
- how minibatch bias and self-interaction affect the endpoint drift gradient;
- whether Muon remains stable with an intermittent distributional loss;
- which adjacent one-step objectives provide useful stabilization without
  compromising the encoder-free or one-NFE claim.

---

## 14. Deep literature audit: what changes after reading the adjacent work

The initial proposal above was internally coherent, but it was not the
strongest mechanism available. Three recent results change the recommendation.

### 14.1 The practical paper drift is a proxy, not a fully balanced flow

The analysis in *On the Wasserstein Gradient Flow Interpretation of Drifting
Models* distinguishes the idealized analytic field from the affinity procedure
actually used by the original image model. The implemented geometric mean of
row and column softmaxes resembles one balancing step, but it does not in
general enforce both prescribed marginals and is not generally the Wasserstein
gradient of a scalar distribution functional.

This matters for the geometry failures observed in this repository. A
row-normalized kernel can spend nearly all its mass on whatever points happen
to be locally close. If generated and real modes are far apart, a narrow kernel
can produce an exponentially small or misleading local signal. It has no
global obligation to allocate transport mass to every target region.

### 14.2 Direct field-energy minimization is not a safe primary objective

The B2 loss minimizes \(\mathbb E_a\|V(a)\|^2\) by differentiating through
the generated particles. Although equality of distributions makes this energy
zero, parameter optimization can lower the empirical energy in other ways:
particles can collapse, spread, or move away from the probes until the local
kernel interaction becomes weak. The positive B2 result remains valuable as a
mechanism test, but it does not make field energy the safest image-training
loss.

For a genuine Wasserstein gradient-flow velocity

\[
V_q(x)=-\nabla_x\frac{\delta F}{\delta q}(x),
\]

the detached drifted-target regression

\[
\mathcal L_{\mathrm{DT}}(\theta)
=\mathbb E_z\left[
  \|G_\theta(z)-
    \operatorname{stopgrad}(G_\theta(z)+\eta V_q(G_\theta(z)))\|^2
  \right]
\]

satisfies, at the ideal population level,

\[
\nabla_\theta\mathcal L_{\mathrm{DT}}
=2\eta\nabla_\theta F(q_\theta).
\]

The stop-gradient is therefore essential. It turns a distributional velocity
into a descent direction for a scalar energy; it is not merely an engineering
trick.

### 14.3 Full Sinkhorn balancing is the best-supported correction

*Sinkhorn-Drifting Generative Models* replaces the one-pass proxy with
two-sided entropic optimal-transport scaling. *One-Step Generative Modeling via
Wasserstein Gradient Flows* (W-Flow) then uses the resulting cross-minus-self
velocity to train a static one-call generator. Its image ablations report that
the fully balanced Sinkhorn velocity is stronger than the original drifting
proxy, MMD, and KL alternatives, and that using an independent second generated
batch for the self term is crucial.

There is an equally important limitation: W-Flow's headline ImageNet system
still uses a pretrained VAE and a pretrained latent feature metric. It is not a
solution to the encoder-dependence question. It does, however, identify the
most credible distributional update to test after replacing the learned
geometry with a fixed injective one.

The revised central question is therefore:

> Can pixel MeanFlow provide the one-step image capability, while a fully
> balanced Sinkhorn drift in a fixed injective multiscale coordinate system
> supplies global coverage without a VAE or learned feature encoder?

## 15. Revised mechanism: pixel MeanFlow plus injective Sinkhorn drift

Call the experimental mechanism **pMF + injective Sinkhorn drift**. This is a
descriptive name, not a claim of a new theorem or established model family.

### 15.1 One-step endpoint law

Keep the pMF network and its full-triangle training exactly as in Sections
1--5. Its deployed endpoint map is

\[
G_\theta(z)=\widehat x_\theta(z,0,1),
\qquad z\sim\mathcal N(0,I),
\]

and the generated law is \(q_\theta=(G_\theta)_\#\gamma\). Inference still
requires exactly one call to \(G_\theta\).

### 15.2 Fixed injective coordinate map

Let \(H\) be a complete orthonormal two-level Haar transform on the image,
including the low-frequency coefficients and **every** detail subband. Let
\(D\) be a frozen diagonal rescaling with strictly positive entries. Define

\[
A=DH.
\]

Because no coefficient is discarded and every scale is positive, \(A\) is
invertible. Thus

\[
A_\#p=A_\#q \quad\Longleftrightarrow\quad p=q.
\]

This is the direct lesson from the identifiability formalization: a convenient
feature space is safe for source-space claims only when it is
measure-determining. Average pooling alone is not injective. A complete Haar
transform is.

The diagonal entries should combine two frozen quantities:

1. a robust training-split scale for each subband, such as RMS or median
   pairwise coefficient distance; and
2. a preregistered positive importance weight for that subband.

All weights must be bounded below by a numerical floor. Coarse bands may be
emphasized, but fine bands may not be zeroed. If all Haar weights are equal,
orthonormality makes the cost exactly raw pixel \(L^2\); nonuniform positive
weights are what create a multiscale Mahalanobis geometry.

This construction is externally encoder-free, not representation-free. It
adds no learned invariance, and it may still be inferior to a semantic encoder
on heterogeneous data. The one-class, 32-by-32 experiment is deliberately
chosen so that this limitation is testable rather than hidden.

### 15.3 Two balanced transport problems

At a correction event, draw independently

\[
z_i,z'_k\sim\gamma,\qquad
x_i=G_\theta(z_i),\qquad
x'_k=G_\theta(z'_k),\qquad
y_j\sim p_{\mathrm{data}}.
\]

The second generated batch \(x'\) is not optional. It is an independent
empirical support for the self interaction. Compute transformed points
\(a_i=Ax_i\), \(a'_k=Ax'_k\), and \(b_j=Ay_j\), and quadratic costs

\[
C^{qp}_{ij}=\tfrac12\|a_i-b_j\|^2,
\qquad
C^{qq}_{ik}=\tfrac12\|a_i-a'_k\|^2.
\]

For uniform empirical masses, solve the two entropic OT problems

\[
\pi^{qp}=\arg\min_{\pi\in\Pi(u_B,u_M)}
  \langle\pi,C^{qp}\rangle
  +\varepsilon\operatorname{KL}(\pi\|u_B\otimes u_M),
\]

\[
\pi^{qq}=\arg\min_{\pi\in\Pi(u_B,u_{B'})}
  \langle\pi,C^{qq}\rangle
  +\varepsilon\operatorname{KL}(\pi\|u_B\otimes u_{B'}).
\]

Both row and column marginals must be enforced. For a row mass \(1/B\), the
conditional barycentric maps in transformed coordinates are

\[
T_p^A(a_i)=\sum_j B\pi^{qp}_{ij}b_j,
\qquad
T_q^A(a_i)=\sum_k B\pi^{qq}_{ik}a'_k.
\]

The empirical Sinkhorn velocity is

\[
V_i^A=T_p^A(a_i)-T_q^A(a_i).
\]

The cross term attracts generated particles toward target mass. The self term
is not a generic hand-written repulsion; it is the debiasing term from the
Sinkhorn divergence and prevents the entropy-regularized cross cost from
preferring an overly diffuse law.

### 15.4 Detached drifted-target loss

Detach \(x'\), \(y\), both transport plans, and the resulting velocity. Only
the primary output \(x=G_\theta(z)\) receives gradients. Use

\[
\mathcal L_{\mathrm{SD}}
=\frac1B\sum_i
\left\|Ax_i-
\operatorname{stopgrad}(Ax_i+\eta V_i^A)\right\|^2.
\]

Since \(A\) is linear and invertible,

\[
A^{-1}V_i^A
=\sum_j B\pi^{qp}_{ij}y_j
 -\sum_k B\pi^{qq}_{ik}x'_k.
\]

Thus the same target may be represented in pixels, but the loss must retain
the \(A\)-geometry if the intended flow lives in the transformed coordinates.
Using raw Euclidean regression after choosing a nonuniform \(A\) would silently
change the parameter-space descent direction.

The full hybrid loss is

\[
\mathcal L
=\mathcal L_{\mathrm{pMF}}
+\lambda_{\mathrm{SD}}\mathcal L_{\mathrm{SD}}.
\]

The roles are complementary:

- pMF supplies dense samplewise supervision over the whole time triangle and
  makes the one-step endpoint map learnable in raw pixels;
- Sinkhorn drift supplies a globally balanced distribution correction at the
  endpoint;
- the fixed invertible transform supplies multiscale conditioning without a
  learned encoder.

No Sinkhorn solve, Haar transform, JVP, real batch, reference bank, or second
generated batch is used at inference.

## 16. Why this is stronger than the initial Laplace B2 proposal

| Question | Laplace B2 field energy | Balanced Sinkhorn drifted target |
|---|---|---|
| Scalar population objective behind update | not guaranteed for the practical differentiated energy | Sinkhorn divergence under the stated OT construction |
| Distant modes | kernel may be nearly flat or exponentially weak | column marginal forces target mass to be represented |
| Bias correction | difference of local normalized means | explicit cross-minus-self debiasing |
| Safe gradient construction | direct differentiation can exploit field weakness | stop-gradient regression follows the WGF energy gradient ideally |
| Batch dependence | simple but locally normalized | more expensive, globally coupled plan |
| Existing image evidence | positive repository mechanism test | strongest recent drifting/WGF ablations |
| Connection to this repository's theorem | direct for the raw population Laplace field | only the injectivity design lesson transfers |

The last row is important. The formalized arbitrary-target Laplace converse
does **not** automatically prove the Sinkhorn-Haar mechanism. The revised model
uses the formalization to avoid a non-injective feature shortcut; its scalar
Sinkhorn guarantees come from the Sinkhorn literature.

Likewise, definiteness of the population Sinkhorn divergence must not be
overstated as a theorem that every finite-particle zero-velocity configuration
is unique. The finite empirical stationarity question is subtler, and the
Sinkhorn-Drifting paper records a remaining finite-sample case. Our experiment
uses finite batches and finitely converged plans, so it is an approximation to
the clean population flow.

## 17. Solver and estimator details that are non-negotiable

### 17.1 Log-domain balancing

Use log-domain Sinkhorn updates. Do not exponentiate an unshifted
\(-C/\varepsilon\) matrix in pixel dimension. Stop only when both marginal
residuals meet a frozen tolerance, for example

\[
\max_i \frac{|(\pi\mathbf1)_i-u_{B,i}|}{u_{B,i}}\le10^{-3},
\quad
\max_j \frac{|(\pi^\top\mathbf1)_j-u_{M,j}|}{u_{M,j}}\le10^{-3},
\]

subject to a preregistered iteration cap. Log the residual and cap-hit rate.
A fixed small iteration count is acceptable only after a preflight shows that
it actually meets the tolerance.

### 17.2 Independent self support

Do not reuse \(x\) as both sides of the self plan and do not treat diagonal
masking as an equivalent repair. W-Flow's reported ablation shows a large
quality loss for same-batch self estimation, while a second independent
generated batch performs much better. Compute \(x'\) with the current model
under `no_grad`; using a lagged EMA model changes the empirical functional and
is a separate ablation.

### 17.3 Quadratic cost

Use the quadratic cost above as the primary mechanism. It is the natural
barycentric WGF construction and is favored by the W-Flow image ablation. A
Laplace or cosine cost is not a harmless swap: it changes both geometry and
velocity.

### 17.4 Frozen, outcome-blind scaling

Estimate Haar subband scales only from the training split, freeze them before
the matched fork, and never tune them on FID or final sample grids. Normalize
the resulting cost so a robust reference statistic, such as the training
median pairwise cost, equals one. Published values such as
\(\varepsilon=0.05\) are meaningful only relative to the authors' feature
scaling and must not be copied blindly.

Choose \(\varepsilon\) in a tiny preflight using only numerical criteria:

- marginal residual and cap-hit rate;
- normalized row entropy;
- maximum conditional transport weight;
- velocity norm and finite-value rate;
- target-column utilization.

The preflight may reject a numerically singular or nearly uniform plan, but it
must not select the value with the best generated images.

### 17.5 Rectangular real support

A generated batch of 64 and an independently sampled real support of 128 or
256 is a reasonable starting point if memory permits. Rectangular balancing
retains uniform marginals and gives the cross plan more target support. The
self plan remains between two independent generated supports. Exact sizes must
be chosen after timing the complete pMF-JVP plus correction step.

## 18. Training schedule and matched causal comparison

### Stage A: capability foundation

Train the one-class raw-pixel pMF model without distribution correction until
it produces recognizable, diverse samples and all preflight gates in Section
10 pass. Published pMF results show that direct pixel prediction and the full
time triangle are essential, but they also expose a major caveat: the strongest
published pMF FIDs use learned perceptual losses. The strict encoder-free MSE
ablation is materially weaker. The foundation should therefore be judged by
the actual one-class pilot rather than the headline pMF number.

### Stage B: cloned continuation

Clone parameters, EMA, optimizer state, data cursor, and RNG state. Run:

| Arm | Loss |
|---|---|
| S0 control | pMF only |
| S1 primary | pMF plus injective-Haar Sinkhorn drift |
| S2 short ablation | pMF plus identity-pixel Sinkhorn drift |
| S3 diagnostic only | pMF plus original Laplace B2 field energy |

S1 versus S0 is the main causal comparison. S2 tests whether the nonuniform
multiscale metric contributes beyond full balancing. S3 connects to the older
repository result but should not consume a full run unless the budget allows.

Use the correction on endpoint samples only. Initially apply it every fourth
update; then benchmark whether every-step correction fits the one-day budget.
The published W-Flow mechanism applies a distribution update continually, so
an extremely sparse correction is not a faithful test. Freeze the cadence
before the matched run and report both update-matched and wall-clock cost.

Calibrate \(\lambda_{\mathrm{SD}}\) by the same auditable gradient-ratio rule
used in B2, but recalibrate because the loss has changed. A reasonable initial
event-level target is

\[
\frac{\|\lambda_{\mathrm{SD}}g_{\mathrm{SD}}\|}
     {\|g_{\mathrm{pMF}}\|}=0.25.
\]

Freeze the resulting coefficient. Log gradient cosine and norm ratio at every
correction event. If strong conflict persists, do not silently add PCGrad;
record it as evidence and run gradient projection or alternating phases only
as an explicitly named rescue arm.

### Stage C: distribution-only diagnostic

If S1 is stable, a short continuation with the pMF loss disabled can test
whether the balanced drift maintains or improves a competent generator by
itself. This is the closest local test of “pure drifting.” It is not required
for the hybrid claim and should not precede S1.

## 19. Pseudocode for one correction event

```python
# Primary endpoint batch: the only generated branch receiving gradients.
x = model(z, t=1.0, h=1.0)

with torch.no_grad():
    # Independent support for q in the Sinkhorn self term.
    x_self = model(z_self, t=1.0, h=1.0)
    y = next_independent_real_batch()

    ax_det = A(x.detach())
    ax_self = A(x_self)
    ay = A(y)

    C_qp = 0.5 * pairwise_sqdist(ax_det, ay)
    C_qq = 0.5 * pairwise_sqdist(ax_det, ax_self)

    pi_qp = log_sinkhorn(C_qp, eps, uniform_rows=True,
                         uniform_cols=True, tol=tol, max_iter=max_iter)
    pi_qq = log_sinkhorn(C_qq, eps, uniform_rows=True,
                         uniform_cols=True, tol=tol, max_iter=max_iter)

    # Row marginal is 1 / B, so B*pi is a conditional distribution.
    T_p = (B * pi_qp) @ ay
    T_q = (B * pi_qq) @ ax_self
    velocity = T_p - T_q
    target_A = A(x.detach()) + eta * velocity

loss_sd = ((A(x) - target_A) ** 2).flatten(1).mean()
loss = loss_pmf + lambda_sd * loss_sd
loss.backward()
```

Implementation assertions must verify:

- `x.grad_fn` exists, while `x_self`, `y`, plans, velocity, and target do not;
- each conditional row of \(B\pi\) sums to one within tolerance;
- both column marginals also pass;
- the two generated noise tensors and batches are distinct;
- no self-diagonal mask is applied;
- exactly one model call remains in the exported sampler.

## 20. What adjacent mechanisms contribute, and why they are not primary

### Sliced-Wasserstein flow

Sliced-Wasserstein flows replace a high-dimensional OT solve with many exact
one-dimensional transports along projections. They provide a global signal,
can be made encoder-free, and are the best rescue if Sinkhorn memory or
quadratic cost is prohibitive. Multiscale/local sliced-Wasserstein image work
also suggests structured projections rather than flat random pixel vectors.

The drawbacks are projection variance, nonsmooth empirical sorting, and the
large number of directions needed in image dimension. Recent W-Flow image
evidence favors full Sinkhorn. Sliced transport should therefore be a planned
fallback, not an additional simultaneous loss.

### MMD and kernel-gradient flows

Characteristic-kernel MMD has a clean population zero set, and MMD
Wasserstein-gradient flows are established in the literature. But finite
bandwidth kernels still suffer from weak interactions between distant modes,
the precise failure that global balancing is intended to repair. W-Flow's
image ablation also reports weaker results than Sinkhorn.

Recent kernel-gradient drifting work broadens the admissible geometries, which
may become useful for non-Euclidean domains. It does not yet provide stronger
encoder-free image evidence than the proposed Sinkhorn route.

### Gradient-flow/KDE interpretations of drifting

Some contemporaneous papers interpret drifting as a KDE-smoothed KL
Wasserstein flow. The later Gretton et al. analysis distinguishes the idealized
analytic algorithm from the practical affinity rule and shows that the latter
does not inherit the claimed scalar WGF interpretation in general. The revised
design therefore uses a transport construction whose scalar functional is
explicit rather than relying on the disputed equivalence.

### Long-short flow maps

The long-short flow-map viewpoint is conceptually compatible with the hybrid:
pMF learns a long interval map, while a distributional velocity corrects its
endpoint. It does not prove that the two losses are jointly convergent, so it
supports the architecture but does not remove the need for the matched fork
and gradient-interaction measurements.

## 21. Theory and claim ledger

| Statement | Status |
|---|---|
| pMF endpoint requires one model evaluation | exact architectural identity |
| MeanFlow regression recovers the ideal average field | population training identity under its assumptions; neural optimization is approximate |
| Sinkhorn divergence has a definite population zero under the standard conditions | established literature result |
| Detached target follows the scalar WGF parameter gradient | ideal population/exact-velocity identity |
| Finite-batch, finite-iteration correction equals that ideal flow | false in general; it is an estimator/approximation |
| Fixed complete Haar coordinates preserve equality of source laws | exact, because the map is invertible |
| The repository's Laplace converse proves the Sinkhorn mechanism | false; only its injectivity lesson is reused |
| The hybrid is externally encoder-free | true if no learned VAE, feature network, perceptual loss, or teacher affects training or selection |
| The hybrid is one-step at inference | true if the exported sampler calls the endpoint network once |
| It is a pure Drifting Model | false while the pMF loss remains active |
| It will match learned semantic features on ImageNet | unknown and unlikely to be shown by the proposed budget |

## 22. Falsification criteria and honest failure interpretations

The experiment should be designed to teach us something even if S1 loses.

1. **Foundation failure.** If MSE-only pMF cannot produce recognizable
   one-class images, the correction is not yet the bottleneck. This would
   confirm that pMF's published perceptual supervision carried more of the
   result than hoped.
2. **Stable but no gain.** If Sinkhorn residuals, coverage, and gradients are
   healthy but S1 ties S0, the pMF endpoint may already saturate the small
   target or the fixed metric may add no useful information.
3. **Identity wins, Haar loses.** The chosen subband weighting distorted the
   geometry; full balancing itself remains viable.
4. **Haar wins, identity loses.** Fixed injective conditioning helped without
   a learned encoder, which is the most interesting mechanism result.
5. **Both collapse.** First audit independent self sampling, plan detachment,
   marginal residuals, and step magnitude. Do not immediately interpret this
   as a failure of population Sinkhorn flow.
6. **Good metrics but memorization.** The one-class data are too small for the
   claim. Nearest-neighbor and train/test duplication veto promotion.
7. **Correction lowers Sinkhorn energy only.** That is a mechanism result, not
   an image-quality result.

The strongest supportable success claim would be:

> On a preregistered one-class pixel task, a fixed, invertible multiscale
> Sinkhorn drift improved a matched one-step pixel MeanFlow generator without
> a VAE, learned feature encoder, perceptual training loss, or extra inference
> evaluation.

## 23. Final recommended implementation order

1. Preserve the existing pMF unit tests and add a single-call inference audit.
2. Implement a complete invertible Haar transform and test numerical inverse,
   energy preservation before weighting, and positive frozen subband scales.
3. Implement rectangular log-domain Sinkhorn with row/column residual tests on
   analytically checkable matrices.
4. Implement the two-batch cross-minus-self velocity and verify conditional
   barycenter weights sum to one.
5. Verify the detached-target gradient against a finite difference of the
   finite-sample Sinkhorn divergence on a tiny deterministic particle system.
   Agreement is expected only to numerical tolerance and with the exact same
   entropic convention.
6. Benchmark a complete pMF step and correction event before fixing batch
   sizes, cadence, and total updates.
7. Run a short outcome-blind numerical preflight to freeze cost scaling,
   \(\varepsilon\), solver tolerance, \(\eta\), and gradient-ratio weight.
8. Train the pMF foundation and require recognizable, diverse images before
   spending the matched continuation budget.
9. Clone exact state and run S0 versus S1. Preserve fixed checkpoints and
   fixed evaluation noise.
10. Run S2 only long enough to attribute any S1 gain to multiscale geometry;
    retain S3 and sliced transport as later diagnostics or rescues.
11. Evaluate only after the run protocol is sealed. Report uncertainty,
    nearest neighbors, diversity, solver diagnostics, wall time, and one-NFE
    verification alongside FID/KID.

This is the most effective mechanism presently supported by the intersection
of the repository and current literature. It is stronger than simply attaching
the original Laplace field to MeanFlow, but it remains a focused experiment,
not a guarantee that fixed low-level geometry can replace semantic encoders at
large scale.

## 24. Additional primary literature used in the deep audit

15. Gretton et al., **On the Wasserstein Gradient Flow Interpretation of
    Drifting Models**. This is the key correction distinguishing the idealized
    field, the practical proxy, and genuine scalar WGFs.
    <https://arxiv.org/abs/2605.05118>
16. He et al., **Sinkhorn-Drifting Generative Models**. This develops the
    two-sided cross-minus-self construction and its population connection to
    the Sinkhorn divergence.
    <https://arxiv.org/abs/2603.12366>
17. Han et al., **One-Step Generative Modeling via Wasserstein Gradient
    Flows**. This is the strongest direct evidence for a static one-call
    generator trained with a fully balanced Sinkhorn velocity.
    <https://arxiv.org/abs/2605.11755>
18. Official W-Flow implementation. The log-domain balancing, detached plans,
    quadratic costs, barycentric update, and independent self-support details
    should be cross-checked against this code before implementation.
    <https://github.com/hanjq17/W-Flow>
19. Feydy et al., **Interpolating between Optimal Transport and MMD using
    Sinkhorn Divergences**. This supplies the foundational definiteness,
    interpolation, and computational results for Sinkhorn divergences.
    <https://proceedings.mlr.press/v89/feydy19a.html>
20. Genevay et al., **Learning Generative Models with Sinkhorn Divergences**.
    <https://proceedings.mlr.press/v84/genevay18a.html>
21. Genevay et al., **Sample Complexity of Sinkhorn Divergences**. This is
    relevant to the finite-minibatch approximation and its dependence on the
    entropic regularization.
    <https://arxiv.org/abs/1810.02733>
22. Arbel et al., **Maximum Mean Discrepancy Gradient Flow**.
    <https://proceedings.neurips.cc/paper/2019/hash/944a5ae3483ed5c1e10bbccb7942a279-Abstract.html>
23. Liutkus et al., **Sliced-Wasserstein Flows: Nonparametric Generative
    Modeling via Optimal Transport and Diffusions**.
    <https://proceedings.mlr.press/v97/liutkus19a.html>
24. Du et al., **Conditional Sliced-Wasserstein Flows**. Its structured local
    and multiscale image projections motivate the proposed fallback, not the
    primary loss.
    <https://proceedings.mlr.press/v202/du23c.html>
25. Nguyen and Ho, **Sliced Wasserstein Estimation with Control Variates** is
    adjacent variance-reduction literature for projection-based rescue work.
    <https://arxiv.org/abs/2305.00402>
26. **A Long-Short Flow-Map Perspective for Drifting Models**. This is a
    conceptual bridge between long-horizon one-step maps and local drift
    velocities, not a proof of the hybrid optimizer.
    <https://arxiv.org/abs/2602.20463>
27. **Kernel-Gradient Drifting Models**. This generalizes drifting geometry but
    does not yet supply stronger encoder-free image evidence than Sinkhorn.
    <https://arxiv.org/abs/2605.10727>
28. Cao et al., **Gradient Flow Drifting: Generative Modeling via Wasserstein
    Gradient Flows of KDE-Approximated Divergences**. Its claims should be read
    together with the later distinction in Gretton et al.
    <https://arxiv.org/abs/2603.10592>
29. **Generative Drifting is Secretly Score Matching**. This supplies another
    interpretation and bandwidth-annealing evidence, but does not remove the
    practical proxy issue above.
    <https://arxiv.org/abs/2603.09936>

All papers in this section are used for mechanism design or caveats. None is
being imported as a theorem about the exact finite-batch implementation without
checking that its hypotheses and algorithm match.
