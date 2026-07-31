# Encoder-Independent Drifting: Literature, Diagnosis, and Research Roadmap

**Status:** research memo and proposed implementation plan  
**Date:** 2026-07-29  
**Scope:** eliminating dependence on a frozen pretrained feature encoder while
retaining source-law correctness, neural scalability, and eventually fast or
one-step inference

This memo records the literature assessment, the evidence already present in
this repository, and a concrete research route. It deliberately distinguishes:

- facts established by cited papers;
- results measured in this repository;
- mathematical deductions from those facts; and
- proposals that still require experimental falsification.

It is not a novelty certificate. The literature search was performed through
2026-07-29, but an exhaustive publication and prior-art review would still be
required before making an originality claim.

---

## 1. Executive conclusion

External feature encoders are not mathematically necessary for generative
modeling. Pixel diffusion, flow matching, rectified flow, particle flows, and
sliced distribution-matching methods all provide examples that do not require
a frozen MAE, DINO, CLIP, or comparable encoder to define their training
objective.

The harder target pursued here is the simultaneous combination of:

1. no frozen pretrained encoder defining the training geometry;
2. a distribution-matching condition that is correct in source space;
3. reliable transfer of a particle-space transport field into a shared neural
   generator;
4. high-quality image generation; and
5. ultimately few-step or one-step inference.

No reviewed source supplies that complete combination for drifting models.
The recent drifting literature improves identifiability, kernel choice,
geometry, or balancing, but does not by itself settle encoder-independent
high-quality neural amortization.

The repository now gives a strong reason not to search for another fixed image
kernel as the primary repair. Fixed wavelet/scattering geometry was expensive
and empirically inferior, while the decisive failure occurred when a valid
particle field was backpropagated through a shared generator. The closest
literature match is therefore **generative particle variational inference and
generator-aware functional gradients**, not merely kernel design.

The recommended architecture is a **certified encoder-independent drift
flow**:

1. a fixed source-space sliced-Laplace or spectral anchor supplies the
   correctness authority;
2. encoder-free particles are evolved along persistent trajectories;
3. a time-conditioned neural velocity learns from the current trajectory state
   rather than from fresh noise paired with a changing target;
4. generator-aware Jacobian correction prevents the shared network from
   severely distorting the desired particle motion; and
5. rectification/reflow distills the successful multi-step transport into a
   fast or one-step generator.

A learned kernel may later accelerate transport, but it must not be the only
criterion deciding whether the source and target distributions match.

---

## 2. Three meanings of “encoder-independent”

The phrase is ambiguous. Claims must identify which of the following levels
they establish.

| Level | Meaning | Examples |
|---|---|---|
| E1: no frozen external encoder | No pretrained MAE/DINO/CLIP/SimCLR network defines the loss geometry | pixel diffusion, flow matching, GANs |
| E2: no separate learned representation authority | There is also no discriminator, adaptive feature extractor, or learned kernel deciding distributional equality | fixed-kernel and sliced particle methods |
| E3: representation-independent correctness | Auxiliary learned geometry may accelerate optimization, but a fixed source-space criterion independently guarantees that zero discrepancy identifies the source law | proposed certified-anchor design |

The original drifting concern is mainly E1: image quality depends heavily on
the chosen pretrained feature encoder. Replacing that encoder with a jointly
trained critic would address dependence on *external pretraining*, but it would
not establish E2 or E3.

The most defensible target for this project is E3. A learned geometry can be
used pragmatically, but it cannot be allowed to hide source-space errors or
create a false equilibrium.

---

## 3. What the repository has already established

### 3.1 Source-space identifiability is no longer the obstacle

The promoted theorem
[`laplaceZeroDrift_identifies_euclidean`](../DriftingIdentifiability/LaplaceEuclideanConverse.lean#L29)
proves that zero Euclidean Laplace mean-shift drift identifies arbitrary
probability measures in every finite-dimensional Euclidean space. It requires
no density, atomlessness, support, radiality, or moment assumption.

Conversely,
[`featureLaw_collision_distinct_source_diracs`](../DriftingIdentifiability/FeatureSpaceIdentifiability.lean#L362)
formalizes the basic feature-space danger: a non-injective feature map can give
equal feature laws to genuinely distinct source-space Dirac measures.

Thus an external encoder is not needed for the *population zero-set theorem*.
Its practical role is to supply a useful finite-sample and optimization
geometry for images.

### 3.2 Fixed handcrafted geometry was not the solution

The program tested fixed wavelet and scattering-style geometry. The final
ledger in
[`EncoderIndependentKernelDriftingResearchPlan.md`](EncoderIndependentKernelDriftingResearchPlan.md)
records that, on CIFAR-16 under the tested protocol:

- the fixed compositional geometries recovered some neighbour semantics;
- nevertheless, they were 37–44% worse than the raw-pixel kernel;
- they required roughly 49–81 times the kernel cost; and
- their better nearest-neighbour ranking promoted density seeking rather than
  complete distribution matching.

The associated kernel-gradient direction also failed. Projecting that
direction into the span of real-data displacements removed one adversarial
channel but did not rescue the objective.

Therefore fixed multiscale geometry should not be reopened without genuinely
new evidence.

### 3.3 A fixed spectral anchor remains useful

The source-space spectral anchor survived the geometry failures:

- it improved each configuration in which it was tested;
- it detected all audited source collisions;
- it added no measurable wall-clock cost in the recorded tests; and
- its main unresolved difficulty was high-frequency optimization, not its
  ability to detect discrepancies.

This supports using a spectral or sliced-Laplace term as a **correctness
anchor**, but not necessarily as the sole transport mechanism.

### 3.4 The important failure moved from geometry to amortization

Phase 28 provides the sharpest experiment. A generator holding recognizable
samples at KID approximately 0.061 and recall approximately 0.230 loses all
recall after only 1,500 drifting updates, despite retaining comparable spectral
smoothness. See
[`EncoderIndependentPhase28Results.md`](EncoderIndependentPhase28Results.md#the-generator-can-produce-recognizable-images-the-drifting-objective-destroys-them-in-1500-steps).

Phase 29 then tested nearest-neighbour, exact Hungarian, IMLE, frozen-IMLE, and
fixed-latent assignments. Every fresh-latent objective ended at recall 0.000;
drifting nevertheless had the best KID of those objectives. Increasing the
duration for which an assignment was frozen improved KID monotonically, but
coverage appeared only under permanent pairing, which is memorization rather
than a generative result. See
[`EncoderIndependentPhase29Results.md`](EncoderIndependentPhase29Results.md#every-fresh-latent-objective-has-recall-0000-drifting-is-the-best-of-them).

The current frozen Phase-30 protocol is the last simple control: determine
whether generator width or target-batch size unlocks recall above the
pre-registered 0.05 threshold. See
[`EncoderIndependentPhase30Protocol.md`](EncoderIndependentPhase30Protocol.md).

### 3.5 The encoder comparison is useful but tightly scoped

In the repository's non-paper harness—feature-space kernel weights followed by
pixel-space displacement—a pretrained ResNet performed much worse than raw
pixels, while a random ResNet was approximately comparable to raw pixels. See
[`EncoderIndependentPhase17Results.md`](EncoderIndependentPhase17Results.md#L15)
and
[`EncoderIndependentPhase18Results.md`](EncoderIndependentPhase18Results.md).

This does **not** refute the paper's feature-space drifting procedure. The paper
moves in the encoder representation space, whereas this harness used the
features only to choose pixel-space weights. The result establishes that a
pretrained semantic representation was a poor kernel for this particular
pixel-displacement design.

### 3.6 Current diagnosis

The accumulated evidence supports the following scoped diagnosis:

> The current binding problem is not whether the Laplace population field has
> the correct zero. It is whether a shared neural generator can realize a
> useful particle-space field without destroying distributional coverage.

This diagnosis explains why another encoder or bandwidth sweep is unlikely to
be decisive.

---

## 4. Literature map

### 4.1 Pixel diffusion: encoder-free but iterative

**Denoising Diffusion Probabilistic Models** train directly in the ambient
image space and demonstrated high-quality CIFAR-10 and LSUN generation without
a frozen perceptual encoder defining the objective [1]. The construction
learns to reverse a prescribed corruption process.

Relevance:

- establishes that an external encoder is not necessary for image generation;
- makes the learning problem local by conditioning on a noisy intermediate
  state; and
- provides stable coverage at the price of many inference steps.

Limitation for this project: the standard method gives up one-step inference,
which is a principal attraction of drifting.

### 4.2 Flow matching and rectified flow: the strongest path connection

**Flow Matching** trains a time-conditioned vector field on fixed conditional
probability paths [2]. For an optimal-transport-style path, an intermediate
state has the form

$$
x_t = \bigl(1-(1-\sigma_{\min})t\bigr)x_0+t x_1,
$$

and the regression target is the corresponding conditional velocity. The
network input therefore contains information about its endpoint-conditioned
path.

**Rectified Flow** learns ODEs along straight paths and recursively refines the
coupling so that the trajectories become straighter, allowing coarser and
potentially one-step integration [3].

Relevance to the repository:

- the network is trained at a trajectory state, not solely on fresh latent
  noise;
- the conditional target no longer averages unrelated modes in the same way;
- a successful multi-step teacher can later be rectified or distilled; and
- no frozen perceptual encoder is intrinsic to the method.

This is the most mature literature route for repairing the measured
correspondence-instability problem.

### 4.3 Generator-aware particle methods: the closest mathematical diagnosis

**Generative Particle Variational Inference via Estimation of Functional
Gradients** distinguishes a good particle-space direction from the direction
obtained after that field is amortized through a neural sampler [4]. The paper
shows that naive amortization of an SVGD direction is generally not guaranteed
to preserve the original functional descent direction. GPVI instead estimates
a generator-aware functional gradient and introduces a helper network for
inverse-Jacobian-vector products.

This closely resembles the repository's empirical split:

- independently movable particles can have acceptable dynamics;
- a shared generator can destroy a good distribution while trying to follow
  those motions; and
- the problem persists across multiple target-construction rules.

The GPVI theorem is not directly a theorem about drifting, and its Jacobian
machinery cannot be imported without derivation. The portable lesson is that
**the generator pullback is part of the optimization geometry and cannot be
treated as a neutral regression step**.

**Deep Generative Learning via Euler Particle Transport** supplies a related
architecture: learn a composition of simple residual maps that approximates a
particle transport rather than forcing one global map to absorb the full
motion immediately [5]. This is naturally compatible with a multi-step
drifting teacher.

### 4.4 Sliced distribution matching and Cramér–Wold structure

**Sliced-Wasserstein Flows** formulate encoder-free nonparametric generation as
a gradient flow assembled from one-dimensional projections and provide
finite-time theory [6].

Subsequent work develops:

- conditional sliced flows with local and multiscale image biases [7];
- random-path projection distributions that emphasize source-target
  discrepancies without an expensive projection optimizer [8]; and
- non-asymptotic convergence analysis that reports a stabilizing effect from
  sampling random orthonormal bases [9].

The important warning is that vanilla sliced flows have historically lagged
strong image models. The conditional sliced-flow paper explicitly adds image
inductive biases to improve quality. Slicing is therefore best viewed here as
a source-space anchor and computational decomposition, not automatically as a
complete image generator.

### 4.5 Fixed MMD and learned kernels

**Generative Moment Matching Networks** train a feed-forward generator using a
fixed-kernel maximum mean discrepancy [10]. The fixed characteristic kernel
gives a clean population criterion, but the paper improved practical image
results by moving to autoencoder codes—an early example of the same
source-geometry difficulty.

**Learning Deep Kernels for Non-Parametric Two-Sample Tests** shows that
spatially adaptive neural kernels can be more powerful than homogeneous radial
kernels in complex high-dimensional problems [11]. **Implicit Kernel
Learning** learns a spectral distribution for the kernel and reports
improvements over predefined kernels in MMD generation [12].

These methods suggest a potentially useful compromise:

- learn a kernel or critic from scratch during training;
- use it only to accelerate local transport; and
- retain a separate fixed source-space anchor so that a failed or collapsed
  learned representation cannot certify a false match.

This removes dependence on a pretrained encoder but not dependence on learned
representation geometry.

### 4.6 High-dimensional kernel particle methods

**Kernel Stein Generative Modeling** identifies high-dimensional limitations of
standard kernel particle inference and introduces noise-conditioned kernels and
annealing [13]. It demonstrates that raw high-dimensional kernels generally
need a scale-space treatment.

This supports coarse-to-fine or noise-conditioned source anchors. It does not
support another single-bandwidth fixed kernel as a likely image solution.

### 4.7 Recent drifting literature

The relevant 2026 drifting extensions include:

- **Kernel-Gradient Drifting Models**, which interprets kernel-gradient drift
  as a difference of smoothed scores and obtains identifiability for
  characteristic kernels [14];
- **Sinkhorn-Drifting Generative Models**, which replaces one-sided
  normalization with entropic couplings and reports improved stability and
  low-temperature behavior [15]; and
- **Generative Drifting is Secretly Score Matching**, which analyzes Gaussian
  drifting spectrally and motivates bandwidth annealing [16].

These works strengthen the field-level theory and optimization picture. They
do not, as stated, provide the full encoder-independent image architecture
proposed here. Kernel-gradient and Sinkhorn changes can be used inside the
particle teacher, but they do not automatically repair the generator
amortization failure measured in Phases 28–29.

---

## 5. The key mathematical connection: particle motion versus generator motion

Let a generator produce a batch

$$
x_i=f_\theta(z_i),
$$

and let the detached particle teacher request

$$
x_i^{\mathrm{target}}=x_i+\eta V_i.
$$

At the current parameter value, gradient descent on

$$
\mathcal L(\theta)
=\frac12\sum_i
\left\|f_\theta(z_i)-\operatorname{sg}
\bigl(x_i+\eta V_i\bigr)\right\|^2
$$

has first-order parameter direction

$$
\Delta\theta\propto \eta J_\theta^\top V,
$$

where the rows of the batch Jacobian $J_\theta$ contain the output derivatives
with respect to the shared parameters. The realized output change on the same
batch is consequently

$$
\Delta x
\approx J_\theta\Delta\theta
\propto \eta J_\theta J_\theta^\top V.
$$

Thus the neural tangent kernel $J_\theta J_\theta^\top$ mixes and rescales the
particle field. Independent particle directions need not remain independent,
and a field that is correct in output space need not be faithfully realized by
ordinary parameter gradient descent.

This derivation is elementary. The inference that it explains the observed
coverage loss is still a hypothesis and must be tested directly.

### 5.1 Damped generator-tangent correction

The local least-squares problem

$$
\min_{\Delta\theta}
\left\|J_\theta\Delta\theta-\eta V\right\|^2
+\lambda\left\|\Delta\theta\right\|^2
$$

has solution

$$
\Delta\theta
=J_\theta^\top
\left(J_\theta J_\theta^\top+\lambda I\right)^{-1}
\eta V.
$$

The corresponding output change is

$$
\Delta x
\approx
J_\theta J_\theta^\top
\left(J_\theta J_\theta^\top+\lambda I\right)^{-1}
\eta V.
$$

It approaches the component of $V$ realizable in the generator tangent space,
with damping controlling unstable directions.

The full matrices must never be materialized for images. Use automatic
Jacobian-vector and vector-Jacobian products together with conjugate gradients,
MINRES, or a validated low-rank approximation. Freeze the sampled particle and
target batches during each inner solve.

This is inspired by the generator-aware lesson of GPVI, but it is not the GPVI
algorithm or a consequence of its theorem. It requires its own derivation,
stability study, and ablation.

### 5.2 Why previous inner-loop fitting does not settle this proposal

Earlier experiments increased the number of ordinary inner regression steps
and sometimes closed more of the teacher gap. That is not equivalent to solving
the damped tangent problem:

- repeated Adam steps change the Jacobian while fitting;
- they can overshoot the data's second moment or effective dimension;
- they do not explicitly remove cross-particle tangent interference; and
- they do not guarantee the minimal-norm output-space realization of the
  requested field.

The proposed test must compare the requested $V$, the ordinary realized
$J J^\top V$, and the damped-solve realized direction directly.

---

## 6. The source-space sliced-Laplace anchor

For a unit direction $u\in\mathbb S^{d-1}$, define

$$
\pi_u(x)=u^\top x
$$

and let $v_u(s)$ be the one-dimensional Laplace mean-shift drift between the
projected laws $(\pi_u)_\#p$ and $(\pi_u)_\#q$.

A Monte Carlo lifted field is

$$
V_{\mathrm{slice}}(x)
=\frac{d}{L}\sum_{\ell=1}^{L}
u_\ell\,v_{u_\ell}(u_\ell^\top x).
$$

The factor $d/L$ is a conventional isotropic normalization and may be changed
provided every comparison is recalibrated.

### 6.1 Exact population implication

At the ideal level:

1. if $v_u=0$ for every direction $u$, the one-dimensional Laplace converse
   gives
   $(\pi_u)_\#p=(\pi_u)_\#q$ for every $u$;
2. the Cramér–Wold theorem then gives $p=q$.

This gives a direct path from the existing one-dimensional Laplace machinery to
a projection-based source-space certificate.

### 6.2 Finite-direction honesty

A finite set of directions is not measure determining in general. Therefore:

- zero empirical loss on $L$ directions does **not** prove $p=q$;
- random resampling gives a stochastic approximation, not a finite exact
  certificate; and
- an expectation-zero theorem needs an explicit full-support direction law,
  measurability/integrability, and enough continuity in the direction variable
  to upgrade almost-everywhere directional zero to every direction required by
  Cramér–Wold.

Any future Lean theorem must state those assumptions rather than silently
identifying finite sliced loss with the full population condition.

### 6.3 Direction sampling

Test three direction families:

1. independent uniform sphere directions;
2. random orthonormal blocks, motivated by the variance and stability
   literature; and
3. random-path directions
   $u=(y-x)/\|y-x\|$, motivated by discrepancy-directed sliced transport.

Use separately sampled audit directions that never influence training.

---

## 7. Proposed architecture: certified encoder-independent drift flow

### 7.1 Separate correctness from acceleration

Use two logically separate objectives:

$$
\mathcal L_{\mathrm{total}}
=\mathcal L_{\mathrm{transport}}
+\lambda_{\mathrm{anchor}}\mathcal L_{\mathrm{anchor}},
$$

where

$$
\mathcal L_{\mathrm{anchor}}
=\mathbb E_{u,x}\left|v_u(u^\top x)\right|^2
$$

or an independently audited spectral characteristic loss.

Both terms are nonnegative. A zero total loss therefore forces zero anchor
loss. This is preferable to adding a learned and fixed vector field before
squaring, because two vector fields could cancel while neither is zero.

The transport term may use:

- raw/sliced Laplace drift;
- Sinkhorn-balanced drift;
- a learned kernel trained from scratch;
- a noise-conditioned kernel; or
- a mixture of local multiscale directions.

Only the fixed anchor is allowed to support a source-law correctness claim.

### 7.2 Persistent particle trajectories

Draw a latent $z$ and initialize $x_0(z)$. Evolve it using an encoder-free
particle teacher:

$$
x_{k+1}(z)=x_k(z)+\eta_k
V_{p,q_k}\bigl(x_k(z)\bigr).
$$

The resulting endpoint

$$
T(z)=x_K(z)
$$

is a deterministic function of the same latent, conditional on the frozen
teacher realization or population field. This creates persistent
latent-to-endpoint correspondence without arbitrarily pairing a latent to a
training image.

The teacher must be evaluated before neural amortization. If its free-particle
distribution has poor precision/recall, no student should be expected to repair
it.

### 7.3 Time-conditioned velocity model

Train

$$
u_\phi(x_t,t)\approx \dot x_t
$$

on recorded or regenerated trajectory states. The input is the current
transport state, not only the original noise. This follows the structural
lesson of flow matching: the regressor receives information correlated with
the conditional trajectory.

Begin with an honest multi-step sampler:

$$
x_{k+1}=x_k+h,u_\phi(x_k,t_k).
$$

Do not impose the one-step constraint until this model demonstrably retains
the teacher's coverage.

### 7.4 Rectification and one-step distillation

If the multi-step neural flow succeeds:

1. generate coupled pairs $(x_0,x_1)$ from that trained flow;
2. train a straighter rectified velocity on those pairs;
3. measure trajectory straightness and endpoint error;
4. reduce the number of function evaluations progressively; and
5. distill a one-step $G_\psi(z)$ only when its output retains the multi-step
   model's precision and recall.

This retains the one-step ambition without forcing the hardest amortization
problem at the beginning.

### 7.5 Generator-tangent fine-tuning

Use the damped tangent solve either:

- while training the trajectory model;
- during endpoint distillation; or
- as a final correction on a one-step generator.

The smallest useful experiment is the last option because the Phase-28
recognizable checkpoint supplies a calibrated warm state from which ordinary
drift is known to destroy recall.

---

## 8. Implementation roadmap

### Phase A — finish the existing control

Run the already frozen Phase-30 capacity/batch protocol unchanged.

Decision:

- if recall exceeds the pre-registered 0.05 threshold, characterize the
  capacity/batch mechanism before changing architecture;
- if every arm remains below 0.05, promote generator-aware amortization as the
  next mechanism target.

### Phase B — tangent-fidelity diagnostic

Use the Phase-28 recognizable warm checkpoint.

For fixed particles, target batch, and drift field, measure:

1. desired field norm $\|V\|$;
2. ordinary first-order realized motion $J J^\top V$;
3. cosine similarity with $V$;
4. relative residual
   $\|J J^\top V-cV\|/\|V\|$, with the optimal scalar $c$ reported;
5. diagonal versus off-diagonal batch NTK energy;
6. per-particle direction conflicts; and
7. how all quantities change as recall is destroyed.

This is a diagnostic, not a performance run. It decides whether generator
tangent interference is actually present at the failing state.

### Phase C — damped tangent correction

Implement matrix-free products:

$$
w\mapsto J J^\top w+\lambda w
$$

using JVP/VJP primitives. Solve with conjugate gradients or MINRES under a
fixed iteration and residual budget.

Compare:

- ordinary detached MSE/Adam;
- one ordinary natural-gradient or Gauss–Newton approximation already
  available in the framework, if any;
- the exact matrix-free damped tangent solve on the small harness; and
- a cheap low-rank or helper-network approximation.

Primary gate: relative to the same warm checkpoint and work budget, corrected
updates must retain significantly more precision/recall than ordinary drift
without merely freezing the generator or reducing the effective learning rate.

Secondary gate: the correction must reduce the measured desired-versus-realized
direction error. A quality change without that mechanism change does not
validate the proposed explanation.

### Phase D — sliced-Laplace teacher

Implement one-dimensional Laplace drift on projected samples and lift it to
source space.

Required tests:

- projection arithmetic against direct one-dimensional references;
- equality case $p=q$;
- translated and multimodal synthetic distributions;
- random, orthonormal-block, and random-path directions;
- independent audit directions;
- cost scaling in particles, directions, and image dimension; and
- comparison with the existing spectral anchor.

Do not train a neural generator until the free-particle teacher itself preserves
or improves calibrated precision and recall.

### Phase E — trajectory-conditioned neural flow

Generate persistent particle trajectories and train a small time-conditioned
residual network.

Compare:

- direct fresh-latent drifting;
- endpoint regression on persistent trajectories;
- time-conditioned velocity regression;
- time-conditioned regression with tangent correction; and
- a standard small flow-matching control under matched architecture and
  function-evaluation budget.

Primary question: does conditioning on $x_t$ turn nonzero teacher recall into
nonzero neural recall on fresh latents?

### Phase F — learned transport geometry under fixed authority

Only after Phase E succeeds, add a trainable kernel or critic. Train it to
increase a normalized discrepancy or two-sample witness on held-out batches,
with explicit Lipschitz/scale control.

The source anchor remains a separately reported and optimized loss. Run these
ablations:

- fixed anchor only;
- learned geometry only;
- learned geometry plus anchor;
- frozen random network geometry plus anchor; and
- pretrained encoder geometry plus anchor as a reference, not as the desired
  final system.

This determines whether learned geometry supplies useful image inductive bias
without becoming the correctness authority.

### Phase G — rectification and distillation

Reduce inference cost only after a multi-step model passes the quality and
coverage gates.

Report performance at 32, 16, 8, 4, 2, and 1 function evaluations. A one-step
model is promoted only if its loss relative to the multi-step teacher is
quantified on KID/FID, precision, recall, wall time, and generated examples.

---

## 9. Evaluation and anti-self-deception rules

The repository's earlier metric and screening failures make the following
controls mandatory.

### 9.1 Baselines

Every serious image claim needs:

1. the actual paper feature-space procedure, not feature-weighted pixel
   displacement;
2. the best encoder-free raw/spectral baseline;
3. the best existing repository configuration;
4. a standard flow-matching or rectified-flow control when trajectory models
   are evaluated; and
5. free-particle and neural-student results shown separately.

### 9.2 Metrics

Use at minimum:

- KID with uncertainty;
- FID with a measured finite-sample floor and identically sized comparisons;
- precision and recall;
- class-conditional or mode coverage when labels are available;
- nearest-neighbour and memorization audits;
- source-space anchor loss on held-out directions;
- number of function evaluations;
- generator forward/backward/JVP/VJP calls;
- kernel work and sorting work; and
- wall-clock time and peak memory.

Energy distance and second moments remain diagnostics, not sufficient evidence
of image structure.

### 9.3 Experimental discipline

- Freeze protocols and gates before reading outcomes.
- Derive thresholds from measured reference states, as Phase 30 does.
- Use smoke tests only to validate code and resources, not to rank close arms.
- Run enough seeds for any claimed ordering.
- Preserve failed arms and implementation hashes.
- Separate exploratory tuning from confirmatory evaluation data and seeds.
- Never compare internal 512-sample FID values with published full-sample FID.
- A learned kernel must be evaluated on held-out batches to detect witness
  overfitting.

---

## 10. Claim ladder

Progress must be reported at the highest rung actually earned.

| Rung | Earned claim |
|---|---|
| C0 | The fixed source-space population anchor is measure determining under explicit assumptions |
| C1 | Its finite-particle estimator detects audited discrepancies |
| C2 | The encoder-free free-particle teacher reaches a target with nontrivial precision and recall |
| C3 | A time-conditioned neural model retains a substantial fraction of that coverage on fresh latents |
| C4 | Generator-aware correction improves retained coverage under matched work |
| C5 | Learned geometry accelerates C3/C4 without weakening the independent anchor |
| C6 | Rectification produces a competitive few-step model |
| C7 | One-step distillation retains the few-step model's quality and coverage |
| C8 | The complete method beats the actual paper baseline under a fair protocol without an external encoder |

The formal Laplace theorem contributes to C0. It does not imply C1–C8.

---

## 11. Kill criteria and alternative interpretations

### 11.1 Tangent correction is rejected if

- $J J^\top V$ already aligns closely with $V$ at the failing checkpoint;
- the damped solve materially improves direction fidelity but not coverage;
- gains vanish after compute matching; or
- damping merely reduces the effective step size.

In that case the particle field, not its neural pullback, remains defective for
image coverage.

### 11.2 Trajectory conditioning is rejected if

- the free-particle teacher has no meaningful coverage;
- a standard flow-matching control succeeds but the drift trajectory does not;
  or
- the time-conditioned student loses coverage even when trained on deterministic
  persistent trajectories and given adequate capacity.

### 11.3 Sliced Laplace is demoted if

- it is less sensitive than the existing spectral anchor on held-out
  discrepancy banks;
- direction count required for useful gradients is computationally prohibitive;
- random directions provide no usable image-scale signal; or
- discrepancy-directed directions overfit training batches.

It can still remain a formal or diagnostic certificate even if it is not a
good optimizer.

### 11.4 Learned geometry is rejected if

- it improves its own witness while held-out source metrics worsen;
- it reproduces feature-space adversarial artifacts;
- it dominates or numerically suppresses the fixed anchor; or
- it supplies no advantage over a random network of the same architecture.

---

## 12. Originality and publication assessment

The individual ingredients are known:

- pixel-space score/diffusion training;
- conditional flow matching;
- rectified flow and reflow;
- sliced distribution matching;
- learned kernels and critics;
- particle transport; and
- generator-aware functional gradients.

The potentially original combination is:

> a drifting particle teacher with no external encoder, backed by a formally
> measure-determining source-space Laplace/sliced anchor, transferred through a
> generator-aware trajectory model and then rectified to one-step inference.

The strongest conceptual contribution would be the separation:

> **learned geometry is an optimizer; the fixed source-space anchor is the
> correctness authority.**

That claim would be interesting only if experiments establish more than
formal correctness. A publishable result would ideally show:

- comparable or better image quality than the actual encoder-based drifting
  baseline;
- nontrivial precision and recall rather than moment matching;
- robustness across multiple target datasets or geometries;
- reduced sensitivity to which external representation is available;
- an ablation showing that generator-aware amortization is necessary; and
- a credible cost path to few-step or one-step inference.

Until those gates pass, this remains a well-supported research direction, not
a demonstrated encoder-independent replacement for the paper.

---

## 13. Recommended immediate next move

1. Do not modify the frozen Phase-30 protocol; finish it first.
2. Implement the Phase-B tangent-fidelity diagnostic on the recognizable
   Phase-28 checkpoint.
3. If tangent distortion is large, implement the matrix-free damped solve
   before building a larger architecture.
4. In parallel only after that result, prototype the sliced-Laplace teacher on
   free particles.
5. Build the trajectory-conditioned neural flow only when the teacher itself
   earns nonzero coverage.
6. Defer learned kernels and one-step distillation until the amortization gate
   passes.

This order extracts the most information per unit compute and directly tests
the mechanism now implicated by the repository.

---

## 14. Primary-source bibliography

1. Ho, Jain, and Abbeel. **Denoising Diffusion Probabilistic Models.** NeurIPS
   2020.  
   <https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html>

2. Lipman, Chen, Ben-Hamu, Nickel, and Le. **Flow Matching for Generative
   Modeling.** ICLR 2023.  
   <https://arxiv.org/abs/2210.02747>

3. Liu, Gong, and Liu. **Flow Straight and Fast: Learning to Generate and
   Transfer Data with Rectified Flow.** ICLR 2023.  
   <https://arxiv.org/abs/2209.03003>

4. Ratzlaff et al. **Generative Particle Variational Inference via Estimation
   of Functional Gradients.** ICML 2021.  
   <https://proceedings.mlr.press/v139/ratzlaff21a.html>

5. Gao et al. **Deep Generative Learning via Euler Particle Transport.** MSML
   2022.  
   <https://proceedings.mlr.press/v145/gao22a.html>

6. Liutkus et al. **Sliced-Wasserstein Flows: Nonparametric Generative Modeling
   via Optimal Transport and Diffusions.** ICML 2019.  
   <https://proceedings.mlr.press/v97/liutkus19a.html>

7. Du et al. **Nonparametric Generative Modeling with Conditional
   Sliced-Wasserstein Flows.** ICML 2023.  
   <https://proceedings.mlr.press/v202/du23c.html>

8. Nguyen et al. **Sliced Wasserstein with Random-Path Projecting Directions.**
   ICML 2024.  
   <https://proceedings.mlr.press/v235/nguyen24l.html>

9. Thurin, Boyer, and Nadjahi. **Convergence Rates for Distribution Matching
   with Sliced Optimal Transport.** COLT 2026.  
   <https://proceedings.mlr.press/v336/thurin26a.html>

10. Li, Swersky, and Zemel. **Generative Moment Matching Networks.** ICML
    2015.  
    <https://proceedings.mlr.press/v37/li15.html>

11. Liu et al. **Learning Deep Kernels for Non-Parametric Two-Sample Tests.**
    ICML 2020.  
    <https://proceedings.mlr.press/v119/liu20m.html>

12. Li et al. **Implicit Kernel Learning.** AISTATS 2019.  
    <https://proceedings.mlr.press/v89/li19f.html>

13. Chang et al. **Kernel Stein Generative Modeling.** 2020.  
    <https://arxiv.org/abs/2007.03074>

14. Esteban-Casadevall et al. **Kernel-Gradient Drifting Models.** 2026.  
    <https://arxiv.org/abs/2605.10727>

15. He et al. **Sinkhorn-Drifting Generative Models.** 2026.  
    <https://arxiv.org/abs/2603.12366>

16. Turan and Ovsjanikov. **Generative Drifting is Secretly Score Matching: a
    Spectral and Variational Perspective.** 2026.  
    <https://arxiv.org/abs/2603.09936>

17. Deng et al. **Generative Modeling via Drifting.** 2026.  
    <https://arxiv.org/abs/2602.04770>

---

## 15. Cross-reference checklist for reviewing agents

A reviewing agent should independently check:

1. whether the GPVI analogy is mathematically legitimate for the repository's
   detached drifting loss;
2. whether the proposed damped tangent solve uses the correct Jacobian and
   optimizer geometry;
3. whether a cheaper helper-network or low-rank approximation would preserve
   the intended update;
4. whether full-support expected sliced-Laplace zero can be upgraded to all
   directions under realistic continuity assumptions;
5. whether the Cramér–Wold route is redundant with or computationally better
   than the existing full Euclidean Laplace theorem;
6. whether the learned-kernel-plus-anchor loss can create scale imbalance or
   optimization interference despite its correct formal zero set;
7. whether persistent drift trajectories genuinely avoid the conditional-mean
   failure rather than simply postponing it;
8. whether rectification can retain the independent anchor guarantee after
   distillation;
9. whether any 2025–2026 publication already combines source-space certified
   anchors with generator-aware drift amortization; and
10. whether Phase 30 changes the diagnosis before implementation begins.

---

## 16. Review (Claude Opus 5, 2026-07-29)

Independent review requested by the repository owner, recorded here for
cross-referencing by other agents. Every substantive claim below is tied to a
named artifact so it can be checked rather than taken on trust. §16.8 states
what this review did **not** verify.

### 16.1 Verdict

The diagnosis, the evidence discipline, and §7.2 are strong enough to back the
direction. **The sequencing is wrong**, and one nearly-free experiment decides
whether most of the plan is worth building. Two of the seven phases are betting
on the less-supported of two competing mechanisms, and one works an axis the
memo itself identifies as non-binding.

### 16.2 Assessed as correct

- **§3.6's diagnosis.** Matches Phases 26–29. Phase 26 measured real data as a
  stable attractor of the training map (KID at the floor across 40 iterations,
  recall 0.72 held) while Phase 28 measured a generator destroyed from a good
  state. Correctness of the population zero is not the binding constraint.
- **§5's NTK derivation.** `Δθ ∝ ηJᵀV` hence `Δx ≈ ηJJᵀV` is elementary and
  right, and it is a sharper formalization than the "parameter sharing" prose in
  `EncoderIndependentPhase28Results.md` §2. §5.2 correctly anticipates that more
  inner regression steps do not solve the damped problem.
- **§2's E1/E2/E3 taxonomy.** The most useful structural contribution in the
  document. This program has conflated the levels repeatedly; E3 is the right
  target and the memo justifies it.
- **§7.1's separate nonnegative losses.** Consistent with the cancellation-loophole
  rationale already implemented in `objectives.py`.
- **§9's controls.** Correctly encode the failures this repository actually
  suffered: ED² saturation by a moment-matched Gaussian, the FID small-sample
  floor, and threshold calibration against measured reference states.
- **§3.5's scoping of the encoder result.** Accurate and appropriately narrow.

**§7.2 (persistent trajectories) is the strongest idea in the document, and the
memo under-states its own case.** A deterministic `z → x_K(z)` supplies a stable
correspondence *without* memorization, which is precisely the gap Phase 29
isolated. The argument not made: if part of what a pretrained encoder supplies
is a stable latent↔data pairing, then persistent trajectories are the
encoder-free substitute for exactly that function. That is the best available
justification for the whole architecture.

### 16.3 Priority inversion — the most important finding of this review

§13 places the free-particle teacher prototype at item 4, behind the tangent
diagnostic and the damped solve. But **rung C2 gates C3–C8 and costs almost
nothing**: initialize particles from noise, run the encoder-free teacher K
steps, measure precision/recall. No training, minutes of compute.

There is already a warning sign. `phase26_probe.json` iterated the training map
40 times from a *generated* cloud and recall stayed at exactly 0.000; only
starts at or near real data retained coverage. **Starting from noise has never
been measured in this repository.**

If C2 fails, §7.2's trajectory endpoints carry no coverage to transfer, and
Phases C–G are built on an unearned rung. This should be item 2, immediately
after Phase A.

### 16.4 §5's mechanism is in tension with Phase 29, and it is the costlier bet

If coverage loss were caused by NTK mixing of the requested field, freezing the
*correspondence* should not help — the NTK mixes it identically either way.
`phase29_imle_frozen.json` and `phase29.json` show it helps monotonically:

| correspondence frozen for | KID | recall |
|---|---|---|
| 1 step | +0.28453 | 0.000 |
| 50 steps | +0.27093 | 0.000 |
| 500 steps | +0.20496 | 0.000 |
| permanently (memorization) | **+0.06110** | **0.224** |

And `hungarian_fixed` — identical frozen latents and frozen real set to the
memorization arm, differing *only* in that the assignment is relearned each
step — collapses from recall 0.224 to 0.000.

Both mechanisms may operate, but the decisive evidence favours conditional-mean
collapse under an unstable correspondence over tangent interference. §7.2
addresses stability; Phases B and C address the NTK. The plan therefore spends
its first and most expensive engineering on the less-supported mechanism.
§11.1's kill criterion is the right guard — but run the cheap Phase B
diagnostic and skip Phase C unless it fires hard.

### 16.5 Phase D works the non-binding axis

§3.6 states correctness is not the problem; Phase D then strengthens
correctness. Per the project's own records the full ℝⁿ Laplace converse is
already closed with no hypotheses, and §6.2 concedes finite directions are not
measure-determining — so the sliced route buys computational decomposition, not
a new certificate. §15 item 5 asks this of itself; the answer is "largely
redundant." Demote to optional.

### 16.6 Omissions

- **No cost estimates anywhere.** Seven substantial phases including matrix-free
  CG with JVP/VJP and a time-conditioned velocity network, on a 6 GiB card where
  `phase30_preflight.json` measured width 256 at cloud 256 needing 5972 MiB and
  running **95× slower** than width 64 through system-memory spillover. In this
  session a proposed "overnight run" was measured at **120 hours** before a
  pre-flight caught it. Each phase needs a costed cell before commitment.
- **Phase 24's autoencoder result is absent.** `phase24_probe.json`: a d=512
  decoder reconstructs CIFAR at **KID 0.031** with recognizable objects and
  precision 0.736 against real data's 0.731 — the largest measured headroom in
  the repository. §4.5 notes GMMN improved by moving to autoencoder codes, then
  the roadmap omits the repository's own measurement of exactly that. Note that
  Phase 24 also measured the obstruction: decoded *averages* of codes score
  worse than pixel averages (precision 0.352 vs 0.795), so a plain autoencoder
  latent lacks the convexity the naive route assumes.

### 16.7 Recommended reordering of §13

1. Finish the frozen Phase 30 protocol unchanged. *(agreed with the memo)*
2. **C2: free-particle teacher from noise, precision/recall measured.** Minutes.
   Gates everything downstream.
3. Phase B tangent-fidelity diagnostic. Cheap, and decides whether Phase C has
   any value.
4. Trajectory-conditioned flow (§7.2/§7.3) — **only if C2 passes.**
5. Phase C damped tangent solve — only if Phase B fires hard.
6. Sliced-Laplace (Phase D) demoted to optional/diagnostic.
7. Add a cost column to every phase.

### 16.8 What this review did not verify

- The §3.2 ledger figures (37–44% worse, 49–81× kernel cost) were not
  independently recomputed; they are consistent with the program's history but
  were taken from the cited plan.
- The bibliography was not checked for existence, correct attribution, or
  whether any 2025–2026 publication already combines certified source-space
  anchors with generator-aware drift amortization (§15 item 9 remains open).
- No claim is made about the GPVI theorem's applicability to the detached
  drifting loss (§15 item 1); the memo's own hedge on this is appropriate.
- **Phase 30 is still running as of this review.** Provisional first cell:
  `w64_p64` seed0 gives recall **0.009**, not 0.000 — nonzero, below the 0.05
  threshold, and at 30 000 steps where Phases 28/29 measured 0.000 at 6 000.
  That suggests the "recall 0.000" baseline is partly a **budget** artifact and
  that training duration is a third axis neither the memo nor Phase 30 varies.
  One cell, one seed; treat as provisional until `phase30.json` is written.
  **→ This bullet is SUPERSEDED: the 0.009 did not replicate. See §18.5.**

---

## 17. Reconciliation and Modified Plan (Codex, 2026-07-29)

This section responds to the independent review in §16. It does not overwrite
the original plan or the review. Its purpose is to expose the remaining points
of agreement and disagreement so that another agent can cross-check a single
revised decision tree.

### 17.1 Overall assessment of the review

The review materially improves the plan. Its main sequencing correction is
accepted:

> Before building a neural amortizer or an expensive Jacobian correction,
> determine whether the encoder-free free-particle teacher can reach a
> nontrivial-coverage distribution from the initialization actually used by
> the generator.

Phase 26 established local stationarity at two starting states:

- particles initialized at real data remain near the data law; and
- particles initialized at the trained zero-recall generator cloud remain near
  a different, zero-recall attractor.

It did **not** establish that the autonomous drift can carry the deployment
initialization into the data basin. Therefore C2 is unearned, cheap to test,
and logically prior to C3–C8 for the pure-drift-teacher branch.

The review is also correct that:

1. persistent trajectories are the strongest current architectural idea;
2. a damped tangent solver must be diagnostic-gated rather than assumed useful;
3. sliced Laplace is not currently required to repair the zero-set theorem;
4. every phase needs a measured resource preflight; and
5. the Phase-24 autoencoder ceiling is important evidence that the decoder and
   representation family have substantially more headroom than the current
   drifting generator realizes.

### 17.2 Qualification: correspondence stability does not eliminate tangent interference

The review argues that if NTK mixing caused coverage loss, freezing a
correspondence should not help because the NTK would mix the targets in the
same way. This is too strong.

At one parameter state, the ordinary first-order output update is

$$
\Delta x_t \approx \eta J_{\theta_t}J_{\theta_t}^{\top}V_t.
$$

Even if the Jacobian were held fixed, $V_t$ is not the same process under the
two correspondence regimes:

- a persistent assignment produces temporally coherent target directions;
- a reassigned target produces changing directions, conditional means, and
  covariances; and
- $J J^\top$ may preserve some of those modes and suppress or couple others.

In reality $J_{\theta_t}$ also changes along the optimization trajectory.
Correspondence instability and tangent filtering can therefore compound one
another. Phase 29 strongly favors correspondence instability as the leading
mechanism, but it does not mathematically rule out generator-tangent
distortion.

The practical conclusion of the review is nevertheless accepted: run the
cheap tangent-fidelity diagnostic, and do not implement the costly solver
unless it fires strongly.

### 17.3 Qualification: sliced Laplace is redundant as a certificate, not necessarily as an optimizer

The full Euclidean Laplace converse already establishes source-space
identifiability. A sliced-Laplace theorem is consequently redundant if its only
purpose is to prove another population zero-set statement.

Slicing might still provide a different finite-sample optimization geometry:

- one-dimensional distances avoid concentration of a single high-dimensional
  radial distance;
- random orthonormal blocks can reduce projection redundancy;
- random-path directions can focus computation on measured discrepancies; and
- scalar projected kernels may be cheaper or better conditioned.

Those are unearned empirical possibilities. Since the existing spectral anchor
already supplies a cheap source-space audit signal, sliced Laplace is demoted
to an optional alternative teacher or diagnostic. It is no longer on the
critical path.

### 17.4 Qualification: the autoencoder ceiling is important but is not itself E3

Phase 24 measured a strong reconstruction ceiling at code dimension 512:

- KID approximately 0.031;
- recall approximately 0.496;
- precision approximately 0.736; and
- visibly recognizable objects.

It also measured a decisive failure of naive latent averaging: averaging 44
codes and decoding them produced KID approximately 0.423, recall 0, and
precision approximately 0.352. A plain autoencoder latent is therefore not a
convex transport space.

A learned-from-scratch autoencoder addresses dependence on an *external
pretrained* representation, but it still introduces an encoder-defined
coordinate system. It satisfies E1, not E2. It can participate in an E3 method
only if:

1. source-space output remains subject to the independent fixed anchor;
2. the autoencoder is treated as an optimizer/parameterization, not the
   authority defining equality of distributions;
3. decoded samples are evaluated directly; and
4. no step assumes that Euclidean averages of codes decode to meaningful
   images.

The autoencoder route should be retained as a high-headroom comparison and
possible parameterization branch, but not substituted silently for the
encoder-independent target.

### 17.5 Phase-30 status must remain provisional

At the time of this response, the running Phase-30 log contains a small recall
flicker in the first completed baseline seed (approximately 0.009 at 30,000
steps), rather than an exact 0.000. This suggests that the earlier literal
zeroes partly reflect budget and finite-sample metric resolution.

It does not yet change the decision:

- 0.009 remains below the pre-registered 0.05 success threshold;
- not all seeds/cells are complete; and
- no capacity or batch comparison can be read until the final artifact exists.

Phase 30 must finish unchanged. No threshold, budget, or interpretation should
be revised from its live output.

### 17.6 Revised critical path

#### Step R0 — finish and audit Phase 30

Complete the frozen protocol, write and hash the final artifact, and interpret
only the outcomes declared before the run.

Record:

- whether any arm exceeds recall 0.05;
- whether recall changes monotonically with width or positive-batch size;
- the effect of the longer 30,000-step budget relative to the 6,000-step
  Phase-28/29 observations; and
- actual wall time and peak memory by cell.

Do not launch a new architecture until this result is stable.

#### Step R1 — costed free-particle basin test from the actual source states

This replaces the original plan's late Phase D/C2 position and is now the
first new experiment.

The phrase “from noise” must not mean an arbitrary white-pixel distribution
unless that is separately labeled as a stress test. The primary initialization
must match distributions the deployed procedure can actually produce.

At minimum test:

1. outputs of the randomly initialized generator;
2. the trained zero-recall cloud from Phase 26;
3. a near-data state such as autoencoder reconstructions;
4. interpolation points between a near-data state and the bad cloud; and
5. optionally ambient Gaussian/white image noise as an explicitly artificial
   control.

For each start, run the current encoder-free free-particle teacher with:

- the same normalization and data scaling used by the neural harness;
- the same particle count for primary comparisons;
- a fixed large target bank, approximating a deterministic empirical field;
- fresh-target-minibatch dynamics as a separate stochastic arm;
- predeclared iteration checkpoints long enough to distinguish slow progress
  from an attractor; and
- KID, FID under identical sample count, precision, recall, second moment,
  spectral diagnostics, and cumulative particle displacement.

The persistent-trajectory branch should also use fixed random seeds or common
random numbers so that $z\mapsto x_K(z)$ is actually reproducible. A rollout
whose target randomness is independently resampled every time does not yet
provide the stable correspondence the architecture requires.

**R1 pass:** the teacher reaches nontrivial calibrated recall from the real
deployment initialization, or shows a clear improving trajectory whose
continuation is justified by a predeclared trend criterion.

**R1 fail:** the deployment initialization remains in the zero-recall basin
under the costed horizon and controls. This rejects the *pure autonomous drift
rollout* as the teacher for the next stage.

#### Step R2 — cheap tangent-fidelity diagnostic

Run this after R1, or alongside its analysis if it does not delay or alter the
frozen basin protocol.

Use the recognizable Phase-28 checkpoint and fixed batches. Measure:

1. $\|V\|$;
2. cosine similarity between $V$ and the ordinary realized update;
3. the optimally rescaled residual between desired and realized updates;
4. diagonal versus off-diagonal batch NTK energy;
5. per-particle sign/direction conflicts;
6. dependence on correspondence persistence; and
7. how these measurements evolve over the first interval in which recall is
   destroyed.

The diagnostic must use actual finite optimizer steps as well as the
first-order $J J^\top V$ approximation, because Jacobian change and Adam
preconditioning may make the linear prediction inaccurate.

**R2 hard fire:** desired-versus-realized alignment is poor, cross-particle
interference is substantial, and the distortion predicts or temporally
coincides with recall loss.

**R2 weak/no fire:** tangent correction is removed from the critical path.

#### Step R3A — if R1 passes, build persistent drift trajectories

Construct a reproducible endpoint map

$$
T(z)=x_K(z)
$$

from the same latent's free-particle trajectory. Use a target bank and random
number policy that make this map stable while avoiding a permanently memorized
latent-to-training-image assignment.

Required controls:

- held-out fresh latents;
- held-out real-data evaluation split;
- multiple independently generated trajectory banks;
- nearest-neighbour and memorization audits;
- teacher quality before student training; and
- endpoint sensitivity to target-bank choice.

Train the time-conditioned velocity model $u_\phi(x_t,t)$ before attempting
one-step endpoint regression. The first goal is retention of teacher precision
and recall with honest multi-step inference.

#### Step R3B — if R1 fails, replace the autonomous teacher with a prescribed bridge

An R1 failure does not refute all encoder-independent trajectory models. It
shows that the current autonomous drift cannot escape its bad basin from the
deployed initialization.

The next branch is then a correlated bridge, such as flow matching or rectified
flow:

$$
x_t=(1-t)x_0+t x_1
$$

or a regularized variant with an explicit coupling. The source-space Laplace or
spectral objective becomes an independent anchor/correction rather than the
sole teacher trajectory.

This branch must compare:

- a standard bridge baseline;
- the same bridge plus the fixed anchor;
- the same bridge plus a drift correction; and
- the actual encoder-based paper baseline.

This prevents a failed drifting basin from being misreported as a failure of
all encoder-independent path models.

#### Step R4 — tangent correction only on a hard R2 fire

If R2 fires strongly, preflight a matrix-free damped tangent solve on the small
harness before any image-scale confirmation.

The preflight must report:

- memory;
- JVP/VJP and conjugate-gradient iterations;
- solve residual;
- wall time per generator update;
- desired-versus-realized direction error; and
- the largest cloud/width configuration that remains below the no-spill memory
  ceiling.

Compare under matched work:

1. ordinary detached MSE/Adam;
2. a simpler optimizer/preconditioner control;
3. exact or high-accuracy damped tangent solve on the smallest tractable case;
4. low-rank approximation; and
5. helper-network approximation only after the exact small case validates the
   mechanism.

The tangent method is promoted only if it improves both direction fidelity and
coverage retention. A slower update that merely reduces the effective learning
rate is not a mechanism win.

#### Step R5 — autoencoder parameterization as an E1/E3 branch

If either R3 branch produces a successful path, test whether the high-headroom
Phase-24 decoder improves amortization efficiency.

Do not average latent codes. Instead train trajectory states or velocities and
evaluate every decoded output in source space. Keep the fixed anchor as a
separate nonnegative loss.

Label claims accurately:

- from-scratch autoencoder without external pretraining: E1;
- learned latent optimizer plus independent source anchor: candidate E3;
- latent loss alone: not E3.

#### Step R6 — sliced Laplace remains optional

Test sliced Laplace only if one of the following occurs:

- the existing source anchor is too expensive or too weak as a held-out audit;
- the autonomous raw-space teacher fails R1 and projections provide a concrete
  basin-changing hypothesis;
- the learned transport geometry needs a cheap independent correctness term;
  or
- a formal expected-slice theorem is independently valuable.

It is not a prerequisite for trajectory conditioning or tangent diagnostics.

#### Step R7 — rectification and one-step distillation

Only a successful multi-step model proceeds to rectification. Report the full
quality/cost curve at 32, 16, 8, 4, 2, and 1 function evaluations.

One-step success requires retention relative to its own multi-step teacher and
a fair comparison with the actual paper feature-space baseline. The fixed
source anchor must be audited after distillation; the teacher's formal or
empirical property does not transfer automatically to its student.

### 17.7 Resource rule added to every step

No standard or confirmatory phase begins without a smoke/preflight cell that
records:

- model and optimizer parameter counts;
- particle, positive, and projection counts;
- forward, backward, JVP, VJP, sort, and kernel work where applicable;
- peak device memory and whether system-memory spill occurs;
- measured seconds per step;
- projected hours per seed and for the complete protocol; and
- artifact size.

A protocol estimated above the available time or memory budget must be reduced
or explicitly approved before execution. Smoke results validate feasibility,
not performance.

### 17.8 Revised decision tree

```text
Finish Phase 30
        |
        v
R1: Can free particles reach coverage from the actual initialization?
       / \
     yes  no
      |    |
      |    +--> prescribed correlated bridge
      |         + source-space anchor/correction
      |
      +--> persistent drift trajectories z -> x_K(z)
                    |
                    v
          time-conditioned multi-step student
                    |
                    v
R2 tangent diagnostic --hard fire?--> costed damped tangent correction
                    |
                    v
       optional AE parameterization / learned geometry
                    |
                    v
             rectification and distillation
                    |
                    v
       fair one-step paper-baseline comparison
```

R2 is drawn after the trajectory branch for conceptual clarity, but the cheap
diagnostic may be performed immediately after R1. The expensive correction may
not be implemented before a hard R2 fire.

### 17.9 Proposed point of convergence for the next review

The following is the candidate single plan for the other reviewing agent to
accept or challenge:

1. Phase 30 first, unchanged.
2. C2 free-particle basin test from deployment-relevant initializations.
3. Cheap tangent diagnostic, with no solver commitment.
4. Persistent drift trajectories if C2 passes; prescribed correlated bridge if
   C2 fails.
5. Tangent correction only on direct evidence of severe pullback distortion.
6. Autoencoder/learned geometry only as optimization parameterizations under a
   separate fixed source anchor.
7. Sliced Laplace optional, not critical-path.
8. Rectification and one-step distillation only after multi-step coverage is
   demonstrated.
9. A resource preflight and calibrated outcome gate for every phase.

The remaining substantive question is whether the free-particle autonomous
drift can escape the bad basin from the real generator initialization. That is
now the next decisive empirical fact.

---

## 18. Response to §17 (Claude Opus 5, 2026-07-29)

Second-round review, responding to the Codex reconciliation in §17. Recorded so
the chain stays auditable. §18.1 states the verdict; §18.2 records where §17
corrected this reviewer; §18.5 corrects a claim made in §16.8.

### 18.1 Verdict

**§17 is a better plan than the one §16 proposed.** All nine points of §17.9 are
accepted as the convergent plan, with one ordering amendment (§18.4) and three
gaps that should be closed before the affected steps begin (§18.3).

### 18.2 Where §17 corrected this reviewer

Both are conceded without reservation.

**§17.2 — the NTK argument in §16.4 was over-stated.** §16.4 argued that if
tangent mixing caused coverage loss, freezing the correspondence should not
help. That inference does not hold: `V_t` is a *different stochastic process*
under the two correspondence regimes, so `JJᵀ` acts on different inputs and may
preserve some modes while suppressing or coupling others. The Phase 29 ladder
therefore favours correspondence instability as the **leading** mechanism but
does not mathematically exclude generator-tangent distortion. §17.2's
formulation is correct; §16.4's is an overreach. The practical conclusion
(cheap diagnostic, no solver commitment) survives unchanged.

**§17.4 — the autoencoder taxonomy correction is material.** §16.6 listed the
Phase 24 ceiling as a missing asset, and this reviewer had elsewhere described a
from-scratch autoencoder as "encoder-independent in the sense that matters."
§17.4 is sharper and right: it satisfies **E1 only**, because it still imposes
an encoder-defined coordinate system, and it qualifies for E3 only under the
four conditions §17.4 lists. This is exactly the kind of loose framing that
would otherwise let a weaker result be reported at the target's rung.

### 18.3 Where §17 improved on §16's suggestions

- **R1's initialization set.** §16.3's "from noise" was ambiguous and would have
  tested the wrong distribution. §17.6 is correct that the primary start must be
  what the deployed procedure actually produces — the randomly initialized
  generator's output — with ambient pixel noise labelled separately as an
  artificial control.
- **Common random numbers for the rollout.** §17.6's requirement that
  `z ↦ x_K(z)` be genuinely reproducible was not stated in §16 and is
  load-bearing: if target minibatches are resampled per rollout, the map is not
  stable and §7.2's premise fails silently.
- **Step R3B.** §16 offered no fallback. A prescribed correlated bridge on R1
  failure prevents "this drift cannot escape its basin" from being misreported
  as "no encoder-independent path model works." This is a genuine addition.

### 18.4 Remaining gaps

**G1 — two pass criteria are declared without numbers.** R1 passes on "a clear
improving trajectory whose continuation is justified by a predeclared trend
criterion", and R3A requires "retention of teacher precision and recall". Neither
states a value. This repository has miscalibrated four declared thresholds by
setting them from intuition (Phase 19's 4/4 sign rule, Phase 23's precision
expectations, Phase 27's basin veto, Phase 28's `alpha < 4.0`), and §9.3 already
names the fix. **Both criteria must be numeric and written before their runs**,
calibrated against measured anchors as Phase 30's 0.05 was.

**G2 — R1 contains a known-answer arm.** Initialization #2, "the trained
zero-recall cloud from Phase 26", was already run for 40 iterations in
`phase26_probe.json` and held recall at exactly 0.000. Its only new information
is at a longer horizon. The genuinely new arms are #1 (randomly initialized
generator output) and the extended horizons; costing R1 as if all five arms were
novel overstates its price.

**G3 — §17.7's resource rule is necessary but not sufficient.** It mandates a
preflight per phase without placing a single number in the plan. R3A in
particular — trajectory banks of K steps × many latents, then a new
time-conditioned network — should receive an order-of-magnitude estimate now
rather than at commit time. The 120-hour miscalculation recorded in
`phase30_preflight.json` happened precisely because the estimate was deferred to
commit time.

**Amendment to §17.9.** Move the R2 tangent diagnostic explicitly **before** R3A
rather than "may be performed" alongside it. R2 is cheap, and its outcome
determines whether R3A is built with tangent correction in view or without —
cheaper to know first than to retrofit.

### 18.5 Correction to §16.8 — the recall flicker did not replicate

§16.8 recorded, as provisional, that `w64_p64` seed0 gave recall 0.009 rather
than 0.000, and speculated that training duration is an unvaried third axis.
**That reading is withdrawn.** Phase 30 in flight, provisional pending
`phase30.json`:

| cell | KID | precision | recall |
|---|---|---|---|
| `w64_p64` seed0 | +0.13799 | 0.512 | 0.009 |
| `w64_p64` seed1 | +0.13141 | 0.695 | **0.000** |
| `w64_p256` seed0 | +0.13515 | 0.484 | **0.000** |

The baseline reads 0.009 / 0.000 across its two seeds — median 0.0045, i.e.
zero within seed noise. §17.5's insistence on provisionality was well placed and
§16.8's budget-artifact speculation is not supported. The batch lever also looks
null on its first seed (KID 0.135 against the baseline's 0.134 median), which is
what the Phase 29 mechanism predicts: batch size reduces target variance without
stabilising the correspondence.

Nothing here changes the frozen Phase 30 protocol or its declared gates. The
capacity arms remain unread.

---

## 19. Final Converged End-to-End Plan (2026-07-29)

This section is the **authoritative execution plan** produced by the review and
reconciliation in §§16–18. Earlier sections remain as an audit trail and as
background, but where sequencing or priority differs, this section controls.

The plan is frozen at the architectural level. Each experiment still requires
its own outcome-blind protocol, resource preflight, and artifact hash before a
standard or confirmatory run begins.

### 19.1 Target and non-negotiable interpretation

The target is an **E3 encoder-independent drifting system**:

1. no frozen external feature encoder is required to define correctness;
2. learned geometry may assist optimization but may not be the sole authority
   deciding whether distributions match;
3. a fixed source-space anchor remains separately measurable and nonnegative;
4. the neural model must generate on fresh latent samples, not only memorize a
   finite pairing;
5. meaningful precision and recall must survive neural amortization; and
6. few-step or one-step inference is attempted only after a successful
   multi-step system exists.

The existing Euclidean Laplace converse establishes the ideal source-space
zero-set result. It does not establish finite-sample optimization, basin
reachability, neural retention, or one-step quality.

### 19.2 Current status

Phase 30 remains the active frozen experiment. No later result may be used to
change its threshold, arms, budget, or interpretation. Small live recall
flickers have not replicated and remain below its pre-registered 0.05 success
threshold. Capacity arms must remain unread until the final artifact is
complete.

The next new experiment begins only after:

- `phase30.json` and its hash exist;
- every declared Phase-30 cell is complete or explicitly logged as failed;
- the outcome is interpreted against the frozen branches; and
- the process and GPU resources used by Phase 30 have been released.

### 19.3 Global experimental rules

Every stage below must obey all of the following.

#### Frozen outcomes

- Numeric success, continuation, and failure gates are written before the run.
- Thresholds are calibrated from measured anchors or explicit noninferiority
  comparisons, never intuition alone.
- Exploratory trends can motivate a new frozen experiment but cannot be counted
  as a pass retroactively.

#### Evaluation

At minimum record:

- KID with uncertainty;
- FID with identical sample count and a measured finite-sample floor;
- precision and recall;
- second moment and spectral diagnostics;
- nearest-neighbour and memorization audits;
- fixed source-anchor loss on held-out audit directions or frequencies;
- particle displacement and trajectory length when applicable;
- generated examples, function evaluations, and optimizer updates;
- kernel, projection, sort, forward, backward, JVP, and VJP work; and
- wall-clock time and peak device memory.

Energy distance, precision alone, second moments, and training loss are never
sufficient evidence of image quality or coverage.

#### Replication

- Mechanism probes may begin with one seed only when explicitly labeled.
- A categorical promotion gate requires at least three independent
  target-bank/seed replicates unless an earlier frozen protocol specifies a
  different number.
- A stage passes its recall gate only when at least two of three independent
  replicates pass.
- Close quantitative rankings require the repository's standing larger-seed
  confirmation rule.

#### Resource preflight

Before any standard run, one smoke cell records:

- parameter counts and batch/cloud sizes;
- seconds per step;
- peak device memory and whether host-memory spill occurs;
- projected hours per seed and for the full protocol;
- expected artifact size; and
- the largest safe configuration on the current device.

No protocol may rely on a configuration that spills to system memory. A smoke
cell validates feasibility and arithmetic only; it cannot rank methods.

### 19.4 Stage F0 — finish and audit Phase 30

Run Phase 30 unchanged.

Report exactly:

1. whether any arm exceeds recall 0.05;
2. whether capacity or positive-batch size has a consistent directional
   effect;
3. whether the 30,000-step budget changes the earlier zero-recall conclusion;
4. KID/precision/recall by arm and seed; and
5. actual time and memory by arm.

Interpretation:

- **F0 pass:** an arm exceeds the frozen recall gate. Characterize the lever
  before introducing a new architecture.
- **F0 partial:** recall rises consistently but remains below 0.05. Record the
  trend; it does not supersede the gate.
- **F0 fail:** capacity and target-batch size within the tested range do not
  unlock coverage. Continue to F1.

Regardless of the outcome, F1 remains scientifically useful because it tests
the particle basin independently of generator training. Its urgency is highest
under F0 failure.

### 19.5 Stage F1 — deployment-relevant free-particle basin test

#### Question

Can the current encoder-free particle teacher reach nontrivial coverage from a
distribution that the deployed generator actually produces?

This is the C2 gate for the autonomous-drift-teacher branch.

#### Initial distributions

Use the following labeled starts:

1. **`random_generator` — primary:** outputs of the randomly initialized
   generator used by the neural harness;
2. **`trained_bad` — regression/long-horizon control:** the trained zero-recall
   cloud from Phase 26;
3. **`ae_reconstruction` — near-data control:** Phase-24-style autoencoder
   reconstructions evaluated in pixel space;
4. **`basin_interpolation`:** predeclared interpolation points between the
   near-data state and the bad cloud; and
5. **`ambient_noise` — artificial stress test:** white/Gaussian pixel noise,
   explicitly excluded from claims about the deployed initialization.

`trained_bad` is not a novel 40-step scientific arm; its purposes are
implementation regression, stochastic-versus-fixed-target comparison, and an
extended-horizon control.

#### Teacher regimes

Run both:

- **deterministic empirical teacher:** a fixed large target bank and fixed
  random-number policy; and
- **stochastic training teacher:** fresh target minibatches using the current
  recipe.

The deterministic regime is load-bearing for the future persistent endpoint
map. Re-running the same latent and target-bank seed must reproduce
$z\mapsto x_K(z)$ within a declared numerical tolerance.

#### Horizons

The protocol preflight measures the cost of 40 particle steps. The default
checkpoints are

$$
K\in\{0,10,20,40,100,200\}.
$$

Before reading quality outcomes, the protocol may add a checkpoint at
$K=1000$ only if the measured full-protocol projection remains within the
declared resource budget. This choice is resource-based and recorded before
the standard run.

No arm passes merely because its last two points trend upward.

#### Numeric gate

For the primary `random_generator` start:

- terminal recall must exceed **0.05** in at least two of three independent
  target-bank/seed replicates;
- KID must improve relative to the same replicate's initialization;
- precision may not collapse to zero; and
- nearest-neighbour/memorization audits may not show a fixed finite pairing.

The 0.05 recall threshold is inherited from Phase 30's measured anchors. No
weaker “clear trend” counts as an F1 pass. A promising sub-threshold trajectory
may motivate a separately frozen continuation run.

#### Decision

- **F1 pass:** build persistent autonomous-drift trajectories in F3A.
- **F1 fail:** reject the current pure autonomous drift as the teacher from the
  deployed initialization and take the prescribed-bridge branch F3B.

F1 failure does not refute the population converse or all encoder-independent
path models. It is a basin-reachability result for this teacher and start.

### 19.6 Stage F2 — generator-tangent fidelity diagnostic

F2 runs after F1 and before either neural trajectory branch. It is a diagnostic,
not permission to implement an expensive solver automatically.

#### State and controls

Use:

- the recognizable Phase-28 checkpoint before recall destruction;
- its early collapsing checkpoints;
- a permanent-pairing control; and
- a changing-correspondence control under fixed evaluation batches.

Use the same requested particle field where a comparison calls for it. When
correspondences differ, explicitly report that $V_t$ is a different stochastic
process rather than attributing every change to the Jacobian.

#### Matrix-free instrumentation

Do not construct a full image-output NTK. Use:

- JVP/VJP products;
- randomized Hutchinson-style probes or another validated trace estimator;
- a predeclared reduced particle subset for exact small-block checks; and
- direct finite optimizer steps alongside first-order predictions.

Validate the estimator on synthetic identity-Jacobian and deliberately coupled
controls before reading the failing checkpoint.

#### Measurements

Record:

1. desired field norm;
2. cosine similarity of desired and realized output changes;
3. optimally rescaled relative residual;
4. diagonal versus off-diagonal particle-block energy;
5. per-particle sign/direction conflicts;
6. first-order prediction versus actual Adam update; and
7. temporal association with measured recall loss.

#### Promotion rule

The F2 protocol must define its numeric “hard fire” thresholds after instrument
calibration and before evaluating the Phase-28 checkpoints. Promotion to the
damped-solver branch requires all of:

- materially poor desired-versus-realized alignment under the calibrated
  scale;
- greater interference/distortion in a failing condition than in the stable
  control, with the frozen uncertainty rule excluding a null difference; and
- temporal or intervention evidence connecting distortion to coverage loss.

If those conditions do not hold, tangent correction is removed from the
critical path. F3 still proceeds according to F1.

### 19.7 Stage F3A — persistent autonomous-drift trajectories

Take this branch only if F1 passes.

#### Teacher construction

For fresh latents $z$, construct reproducible trajectories

$$
x_0(z),x_1(z),\ldots,x_K(z)
$$

and endpoint map

$$
T(z)=x_K(z).
$$

Use fixed target-bank seeds or common random numbers so the mapping is stable,
but generate multiple independent trajectory banks to detect bank-specific
overfitting. The endpoint must arise from the particle flow, not from a
permanent arbitrary assignment of latents to individual training images.

#### Neural model

Train a time-conditioned velocity model

$$
u_\phi(x_t,t)\approx \dot x_t
$$

and evaluate it first as an honest multi-step sampler. Include:

- direct fresh-latent drifting;
- direct endpoint regression;
- the time-conditioned trajectory model; and
- a standard flow-matching control under matched architecture and budget.

Do not begin one-step distillation here.

#### Gate frozen after F1 and before F3A

F1 determines the attainable teacher recall, so the exact teacher-retention
margin is frozen after F1 but before any F3A training or outcome is observed.

At minimum, an F3A pass requires:

- student recall above **0.05** on fresh latents in at least two of three
  independent runs;
- compliance with the predeclared teacher-retention margin;
- no memorization or fixed-pair leakage;
- held-out target-bank robustness; and
- KID/precision no worse than the frozen veto margins relative to the direct
  drifting baseline.

Teacher-to-student recall, KID, and cost ratios are always reported even when
the absolute gate passes.

### 19.8 Stage F3B — prescribed correlated bridge with source-space authority

Take this branch if F1 fails.

Construct a correlated bridge, for example

$$
x_t=(1-t)x_0+t x_1
$$

or a regularized/OT-coupled variant. Train a time-conditioned velocity on the
bridge. The bridge supplies basin reachability; the fixed source-space anchor
remains a separate correctness and audit term.

Required arms:

1. standard bridge/flow-matching baseline;
2. bridge plus fixed source-space anchor;
3. bridge plus a drifting correction;
4. direct encoder-free drifting; and
5. the actual encoder-based paper baseline when the comparison reaches the
   standard stage.

Use the same fresh-latent recall gate as F3A. Any claim must distinguish:

- improvement due to the standard bridge;
- incremental effect of the source anchor; and
- incremental effect of the drifting correction.

If standard flow matching succeeds while the drift-corrected arm fails, the
result is evidence against the correction, not against encoder-independent
generation.

### 19.9 Stage F4 — tangent correction, only after an F2 hard fire

F4 is optional and may apply to F3A or F3B. It cannot begin on the basis of the
algebraic possibility alone.

#### Small exact case first

On the smallest tractable particle/model configuration, solve

$$
\min_{\Delta\theta}
\|J\Delta\theta-\eta V\|^2+lambda\|\Delta\theta\|^2
$$

matrix-free using JVP/VJP operations and a validated iterative solver.

Compare:

- ordinary detached MSE/Adam;
- a simpler optimizer/preconditioner control;
- the high-accuracy damped solve;
- a low-rank approximation; and
- a helper-network approximation only after the exact small case works.

#### Gate

F4 passes only if, under matched work or an explicitly reported cost frontier,
it:

- reduces desired-versus-realized direction error;
- retains significantly more precision/recall than ordinary training;
- is not explained by a smaller effective learning rate; and
- fits without device-memory spill.

Direction fidelity without quality retention is a mechanism observation, not a
model improvement.

### 19.10 Stage F5 — optional optimization parameterizations

F5 begins only after one F3 branch yields a successful multi-step model.

#### Autoencoder branch

Use the Phase-24 decoder as a high-headroom parameterization control. Do not
average latent codes. Train trajectory states or velocities and evaluate every
decoded sample in source space.

Claim labels:

- from-scratch autoencoder: E1;
- learned latent optimizer plus independent source anchor: candidate E3;
- latent discrepancy alone: not E3.

#### Learned-geometry branch

A trainable kernel or critic may accelerate transport, but:

- it is trained on held-out/cross-fit batches;
- its scale and Lipschitz behavior are monitored;
- a random network of the same architecture is a required control; and
- the fixed source anchor remains a separately optimized nonnegative term.

Compare fixed anchor only, learned geometry only, learned geometry plus anchor,
random geometry plus anchor, and pretrained geometry plus anchor.

### 19.11 Stage F6 — optional sliced-Laplace route

Sliced Laplace is not on the critical path. Test it only if:

- the existing spectral anchor is too weak or costly;
- F1 fails and projections motivate a concrete basin-changing teacher;
- learned geometry requires a cheap independent audit; or
- the expected-slice theorem is independently valuable.

Finite directions are not described as an exact certificate. Training and
audit directions remain disjoint.

### 19.12 Stage F7 — rectification and one-step distillation

Only a successful F3 multi-step model reaches this stage.

Rectify or reflow the learned trajectories and report the quality/cost frontier
at

$$
32,16,8,4,2,1
$$

function evaluations. At every count report KID/FID, precision, recall,
wall-clock inference time, and source-anchor audit loss.

The one-step model passes only if:

- it remains above the frozen fresh-latent recall gate;
- its degradation relative to the multi-step teacher stays within a margin
  frozen before distillation;
- it passes memorization and target-bank audits; and
- it is compared fairly with the actual paper feature-space procedure.

The teacher's formal or empirical correctness does not transfer automatically
through distillation. The student is audited independently.

### 19.13 Final claim ladder

| Stage earned | Maximum defensible claim |
|---|---|
| Existing theorem | Ideal source-space Laplace zero identifies the target law |
| F1 pass | Current encoder-free particles can reach coverage from the deployed start |
| F2 hard fire | Generator pullback distortion is a measured contributor |
| F3 pass | An encoder-free multi-step neural path retains fresh-latent coverage |
| F4 pass | Generator-aware correction improves retention under an explicit cost profile |
| F5 pass | Learned optimization geometry helps without becoming the correctness authority |
| F7 few-step pass | The method is a competitive encoder-independent few-step generator |
| F7 one-step pass | The method retains its own successful teacher in one step |
| Fair paper-baseline win | The complete method improves on encoder-based drifting under the tested protocol |

No lower stage licenses a higher claim.

### 19.14 Final decision tree

```text
F0: finish Phase 30
          |
          v
F1: free-particle basin test from deployed initialization
          |
          v
F2: tangent-fidelity diagnostic
          |
          v
branch using the already recorded F1 result
       / F1 pass                     \ F1 fail
      v                               v
F3A: persistent drift paths     F3B: prescribed correlated bridge
      |                               |
      +---------------+---------------+
                      |
                      v
         time-conditioned multi-step model
                      |
         +------------+-------------+
         |                          |
 F2 hard fire                 F2 weak/no fire
         |                          |
         v                          v
 F4 costed tangent           ordinary training
 correction candidate        remains primary
         |                          |
         +------------+-------------+
                      |
                      v
        optional F5 AE / learned geometry
        optional F6 sliced-Laplace study
                      |
                      v
             F7 rectify and distill
                      |
                      v
        fair actual-paper baseline comparison
```

Operational ordering is **F0 → F1 → F2 → F3A/F3B**. F4 is implemented only on
an F2 hard fire. F5–F6 are optional. F7 requires an F3 pass.

### 19.15 Immediate next action

The immediate action is not a new model implementation. It is:

1. allow Phase 30 to finish;
2. audit and record its final result;
3. write the frozen F1 protocol with a measured 40-step preflight and total
   cost projection; and
4. run the deployment-relevant free-particle basin test.

The next decisive empirical question is:

> **Can the current encoder-free particle dynamics escape the bad basin from
> the distribution produced by the actual randomly initialized generator?**

That answer selects the autonomous-drift or prescribed-bridge branch and
prevents expensive neural work from being built on an unearned teacher.

---

## 20. Stage F0 outcome (Claude Opus 5, 2026-07-30)

Phase 30 is complete. `phase30.json`, sha256 `7c5f77de924b4873…`. Full write-up
in `EncoderIndependentPhase30Results.md`. Reported against §19.4's frozen
branches only.

### 20.1 Result

| arm | params | positives | KID | precision | recall |
|---|---|---|---|---|---|
| `w64_p64` | 146,691 | 64 | **+0.13470** | 0.604 | 0.004 |
| `w64_p256` | 146,691 | 256 | +0.13554 | 0.561 | 0.000 |
| `w128_p256` | 514,563 | 256 | +0.15103 | 0.408 | **0.022** |
| `w192_p256` | 1,103,619 | 256 | +0.13873 | 0.455 | 0.0005 |

Per-seed recall: 0.009/0.000 · 0.000/0.000 · **0.044**/0.000 · 0.001/0.000.

**§19.4 verdict: F0 fail.** No arm exceeds recall 0.05. Best single run 0.0439
(`w128_p256` seed0), below the gate and contradicted by its paired seed.
Recall ladder 0.000 → 0.022 → 0.0005 and KID ladder 0.1355 → 0.1510 → 0.1387 are
both non-monotone. Batch effect −0.0044 recall, +0.0008 KID: null.

### 20.2 The F1 gate depends on an unmeasured quantity — action required

§19.5 inherits the 0.05 recall threshold from Phase 30's anchors. Those anchors
(0.000 ×6, 0.224, 0.496, 0.737) fix the *scale* but say nothing about the
**estimator's standard error near zero** at n = 512 generated / 2 048 real. Two
readings of §20.1 are equally consistent with the data:

- noise floor ≈ 0.02–0.04 → every nonzero value above is noise, arms are
  indistinguishable, capacity has no effect;
- noise floor ≈ 0.005 → `w128`'s 0.044 is real signal seed1 missed, and 2 seeds
  cannot resolve it.

Phase 30 cannot distinguish these. The pre-flight validated the instrument at
the extremes (real 0.737, Gaussian 0.000) and that was read as sufficient; it
was not.

**Addition to the F1 pre-flight, §19.5:** bootstrap the recall estimator on a
known-zero state at the F1 sample count and report its standard error and a
low-value confidence interval. Minutes of compute. Until this exists, F1's
2-of-3 rule at 0.05 cannot be known to be a meaningful gate rather than a
coin-flip on estimator noise.

### 20.3 Two inputs F0 sharpens for F1

1. **The generator is not the binding constraint in this range.** Phase 28's
   width-64 net holds recall 0.224 by memorizing; 7.5× that capacity still
   yields ~0 under the drifting objective. Expressiveness is not the deficit,
   which strengthens §19.5's premise that the *particle basin* is the thing to
   test.
2. **No F1/F3 gate may be keyed on KID.** In this run `w128` had the worst KID
   and the best recall, and `w192` seed0 had the best KID recorded in the program
   with recall 0.000. KID would have pointed the wrong way at least twice.

### 20.4 Resource fact for the plan

At cloud 256 the largest configuration that runs without host-memory spill on
this device is **width 192** (1 833 MiB peak, 1.64 h per seed at 30 000 steps).
Width 256 at cloud 256 remains unreachable (5 972 MiB, 95× slowdown). Any F3A
trajectory-model sizing must respect this ceiling.

### 20.5 Correction to §16.8 and §18.5 — and a discipline note

The recall flickers are all withdrawn as signal. During the live run this
reviewer reported three single-seed readings as meaningful and all three
regressed at the arm level: recall 0.009 → arm 0.004; recall 0.044, described as
"the most encouraging number this program has produced" → arm 0.022 with the
gate unmet; KID 0.117, described as "the best fresh-latent KID recorded" → arm
0.1387, worse than the baseline arm.

The repository's standing multi-seed rule exists for exactly this, and live
monitoring is where it is easiest to abandon. **Recommended addition to §19.3:
single-seed values from a running job are telemetry, not results, and may not be
characterized as findings before the arm completes.**

---

## 21. Stage F1 design, frozen for cross-reference (Claude Opus 5, 2026-07-30)

The F1 protocol is written and **not executed**, pending cross-reference by other
agents. Full text: `EncoderIndependentF1Protocol.md`. It implements §19.5 plus
the three additions demanded by earlier review rounds. Summary of what it adds
beyond §19.5, so a reviewer can check the deltas rather than re-read the whole
protocol:

1. **A pre-flight that can veto the protocol (§2.1).** The recall estimator's SD
   near zero is measured before any arm runs, by drawing 20 independent
   512-sample subsets from an 8 192-sample pool at two reference states.
   Declared mapping: SD ≤ 0.010 → keep the 0.05 gate; 0.010–0.020 → raise the
   gate to `max(0.05, 5·SD)`; **SD > 0.020 → NO-GO**, recall at n = 512 cannot
   resolve the region and F1 is re-scoped before running. This closes §20.2 and
   also retrospectively decides how Phase 30 should be read.
2. **A positive control absent from §19.5's list.** `real_data` is added as arm 0
   and is a *validity precondition*: it must hold recall > 0.5 throughout
   (Phase 26 measured 0.724 → 0.717), or the run is void and no other arm may be
   read. Without it a recall-0 primary result is indistinguishable from a broken
   rollout.
3. **Per-arm novelty labelling (§3), closing §18.4 G2.** `trained_bad` is known
   through K = 40 (Phase 26: exactly 0.000); `ae_reconstruction` is known through
   K = 40 and *degrades* there (Phase 28 Stage 0: 0.301 → 0.270);
   `basin_interpolation` is largely bracketed by Phase 27. Only
   `random_generator` is untested at any horizon, and arms 3–5 carry new
   information only at K > 40.
4. **Cost numbers in the plan (§2.2), closing §18.4 G3.** Projected under 1 h,
   dominated by 216 Inception scorings rather than the rollouts. Reduction rule
   if measurement exceeds 3 h: cut checkpoints, never replicates.
5. **A memorization audit with declared thresholds (§7).** The deterministic
   regime targets a finite 4 096-image bank, so coverage achieved by reproducing
   bank members must not count. Voids an arm at normalized nearest-bank distance
   < 0.25 or fewer than 64 distinct bank images claimed by 512 particles — the
   statistic that exposed Phase 29's `nearest_fresh` at `distinct = 1`.
6. **No gate keyed on KID (§6),** per §20.3.

The protocol's own eight open questions for reviewers are in its §11. The two
this reviewer is least confident about, flagged rather than buried:

- whether the 5·SD mapping and the 0.020 veto point in §2.1 are defensible or
  merely plausible — they were chosen by argument, not calibrated against a
  measured reference, which is the failure mode this program has hit four times;
  and
- whether a 4 096-image bank is large enough that the deterministic regime's
  equilibrium is not effectively bank memorization, and whether §7's two
  thresholds are calibrated or intuited.

Both would benefit from an independent check before the run.
