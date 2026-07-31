# Anchored coherent transport research plan

## Status

This is the live research plan for the next post-PSQT model program.

Implementation update (2026-07-24):

- the inherited Stage 1A/1B effort was audited and its promotion claims were
  withdrawn; see
  [`coherent_transport/ImplementationAudit.md`](coherent_transport/ImplementationAudit.md);
- the EST coupling and sparse balanced matching primitives are now verified;
- the corrected route audit is recorded in
  [`coherent_transport/BalancedPlanAuditResults.md`](coherent_transport/BalancedPlanAuditResults.md);
- the first neural-retention and hybrid-route development results are in
  [`coherent_transport/RouteAmortizationDevelopmentResults.md`](coherent_transport/RouteAmortizationDevelopmentResults.md);
- pure sliced-consensus routing was rejected; the current development
  candidate uses EST only to form a sparse candidate graph and then minimizes
  Euclidean route cost on that graph;
- no confirmation, paper-performance, higher-dimensional, or encoder-robust
  claim has been promoted.

No success claim is made by this document. It records the hypotheses,
implementation order, evaluation protocol, promotion gates, and fallback
decisions that must be settled before another model is called an improvement.

The immediate objective is not to maximize one aggregate metric on the
existing synthetic registry. It is to determine whether the strongest
reusable part of PSQT--persistent information about missing and excess
mass--can be combined with a coherent multidimensional transport plan without
destroying the local support geometry learned by the drifting model.

## 1. Decision

The next candidate will be an **anchored, geometry-aware coverage repair
model**.

The key design change is:

> PSQT will decide *where and how much mass needs intervention*. It will no
> longer independently back-project one-dimensional rank corrections and use
> their average as the final particle displacement.

The candidate has five conceptual parts:

1. a local geometry learner, initially the official-style drifting baseline;
2. a persistent KLL/quantile controller that detects distributional deficits;
3. a coherent sliced-plan or sparse partial-transport planner;
4. persistent or discrete neural routing so incompatible destinations are not
   averaged together; and
5. a fixed injective anchor discrepancy, alongside any optional learned
   semantic encoder.

This program should be abandoned or sharply simplified if the coherent
particle-level teacher cannot simultaneously retain PSQT's rare-mode benefit
and match the paper baseline's support precision on structured two-dimensional
targets.

## 2. Why a new architecture is needed

### 2.1 What PSQT established

The repository has established three useful facts.

First, persistent quantile state is valuable. In one dimension,
Persistent Quantile Transport improved 57 of 60 target/initialization cells
under its confirmatory protocol. In the free-particle two-dimensional
experiments, the KLL PSQT accumulator recovered rare modes and approached the
pooled-quantile ceiling with much less stored target state.

Second, a conditioned neural map can absorb part of this signal. The confirmed
conditioned model beat the registered repository paper port in both ED2 and
SW1 on 62 of 64 synthetic target instances.

Third, the official-style checkerboard and Swiss-roll demonstrations exposed
a different failure axis. Good aggregate projected and energy statistics did
not imply that the generated particles lay on the target support. The current
candidate could improve global mass allocation while increasing bridge mass,
cell leakage, or manifold blurring.

These findings are not contradictory. They say that PSQT is a strong
**coverage statistic** but its current vector field is not a reliable
**geometry-preserving joint transport**.

### 2.2 Structural defect in the current correction

For generated particles \(x_i\), directions \(u_\ell\), and independently
rank-assigned target quantiles \(q_{\ell,\pi_\ell(i)}\), the current correction
has the form

\[
g_i =
\frac{d}{L}
\sum_{\ell=1}^{L}
\left(
q_{\ell,\pi_\ell(i)}-\langle x_i,u_\ell\rangle
\right)u_\ell .
\]

The scalar destinations for different directions need not correspond to one
common target sample. A particle can be assigned to one target component in
one projection and another component in a second projection. The
back-projected average may then lie between both components.

Two neural effects make this worse:

1. re-ranking fresh cohorts changes the teacher assignment discontinuously;
2. a single MSE-trained network averages incompatible destinations.

This gives a specific causal hypothesis for bridges and diffuse modes. The
next program must test this hypothesis directly instead of continuing to tune
the old correction.

### 2.3 Encoder dependence

The formal feature-space results distinguish equality of feature pushforwards
from equality of source distributions. A non-injective feature map can make
distinct source distributions indistinguishable. Therefore a feature-only
empirical success cannot establish that the source distribution was matched.

At the same time, the paper and recent large-scale drifting methods use
pretrained features because raw high-dimensional kernels can become flat or
statistically inefficient. The practical goal is consequently not to forbid
learned features. It is to prevent an unverifiable learned encoder from being
the only authority.

The candidate will use:

- a fixed injective anchor branch for an auditable source-space signal; and
- an optional semantic branch for efficient perceptual routing.

These roles must remain separately measurable.

### 2.4 Repository evidence consumed by this plan

This roadmap is based on the following consumed results and diagnostics:

- [`PQTConfirmatoryResults.md`](PQTConfirmatoryResults.md): the successful
  one-dimensional persistent-quantile result;
- [`PSQTAccumulatorConfirmatoryResults.md`](PSQTAccumulatorConfirmatoryResults.md):
  the free-particle two-dimensional KLL/PSQT result;
- [`NeuralPooledRankPhase1Results.md`](NeuralPooledRankPhase1Results.md): the
  first neural-amortization gap and free-particle ceiling;
- [`NeuralConditionedTransportConfirmatoryResults.md`](NeuralConditionedTransportConfirmatoryResults.md):
  the 62/64 result and its explicitly limited synthetic scope;
- [`AdaptiveRolloutConfirmationResults.md`](AdaptiveRolloutConfirmationResults.md):
  evidence that repeated rollout helps only dimension-adaptively;
- [`ProjectionKernelOptimizationConfirmationResults.md`](ProjectionKernelOptimizationConfirmationResults.md):
  the active-direction and kernel-compression cost findings;
- [`ConditionedTransportRepairFailureDiagnosis.md`](ConditionedTransportRepairFailureDiagnosis.md):
  the rare-core, bridge, and neural-retention diagnosis;
- [`TailSafetyRepairResults.md`](TailSafetyRepairResults.md): the negative
  result for rank ordering and myopic per-particle safety; and
- [`../DriftingIdentifiability/FeatureSpaceIdentifiability.lean`](../DriftingIdentifiability/FeatureSpaceIdentifiability.lean):
  the formal pushforward/source-law boundary and quantitative lifting
  interface.

Terminology in this document follows the repository convention: **PQT** is
the one-dimensional persistent quantile transport mechanism; **PSQT** is its
projected/sliced higher-dimensional descendant.

## 3. Research questions and hypotheses

### RQ1: Is independent sliced assignment the active geometry bottleneck?

**H1.** Replacing the averaged projected correction with a valid joint
coupling will reduce leakage and bridge occupancy on checkerboard, Swiss roll,
pinwheel, rings, and moons without materially worsening ED2 or SW1.

**Falsification.** If a coherent sliced plan does not improve support
precision at the free-particle level, joint inconsistency is not the primary
bottleneck and the PSQT motion branch should be retired.

### RQ2: Should persistent quantiles control all particles or only deficient
mass?

**H2.** A partial-transport intervention selected by persistent deficits will
preserve local geometry better than moving every particle with a global
quantile correction.

**Falsification.** If full coherent transport consistently outperforms partial
transport at equal cost and support precision, the controller is unnecessary
and should be removed.

### RQ3: Is neural route averaging the main amortization loss?

**H3.** Persistent latent cohorts or discrete route heads will retain more of
the free-particle teacher's rare-core mass and support precision than a single
MSE transport head.

**Falsification.** If the route-aware model does not improve teacher retention
under matched capacity, the dominant problem is teacher quality, optimization,
or generator topology rather than label switching.

### RQ4: Can an explicit anchor reduce reliance on the semantic encoder?

**H4.** An anchor-plus-semantic model will degrade substantially less than a
feature-only model when the semantic encoder is weakened, randomized, or made
intentionally non-injective.

**Falsification.** If the anchor branch has no measurable effect under encoder
stress, it is either too weak at the chosen sample size or incorrectly
implemented. It must not be retained merely for rhetorical value.

### RQ5: Can the additional coordination be made computationally competitive?

**H5.** Active KLL directions, a sparse sliced candidate graph, and partial
transport can recover most of a full Sinkhorn teacher's quality at materially
lower projection, kernel, and memory cost.

**Falsification.** If the sparse candidate requires nearly dense transport to
pass the quality gates, report that honestly and compare directly with
W-Flow/Sinkhorn-style methods rather than claiming an efficiency advantage.

## 4. Proposed model

### 4.1 Target-data partitions

Every registered experiment will create disjoint, frozen target pools:

1. **atlas pool:** builds persistent quantile/KLL summaries;
2. **planning pool:** supplies target samples and target identities to the
   joint transport planner;
3. **controller pool:** evaluates proposed interventions and support guards;
4. **evaluation pool:** used only for final metrics.

No final evaluation point or evaluation projection may influence direction
selection, route construction, gain selection, bandwidth selection, stopping,
or checkpoint selection.

The KLL atlas is not enough to define a joint plan because it discards target
sample identity. The implementation must retain a planning reservoir or
weighted coreset in addition to the atlas.

### 4.2 Geometry learner

The first implementation uses the official-style paper drift as the local
geometry learner. This isolates the contribution of the new controller and
planner.

Its responsibilities are:

- form local clusters and manifolds;
- refine within-mode density;
- provide a conservative default update where no global deficit is detected.

PSQT must not perform a large global warm start before this branch has learned
meaningful geometry. The registered order should be either:

1. local burn-in, then interleaved repair; or
2. local update first within every macro-cycle, then a bounded repair.

Both should be compared in development. The selected order is frozen before
confirmation.

### 4.3 Persistent coverage controller

For each active direction \(u_\ell\), estimate generated and target CDFs from
the current generated cohort and persistent target atlas. Define a
directional deficit signal, for example

\[
\Delta_\ell(t)=F_{G,\ell}(t)-F_{T,\ell}(t).
\]

The sign and magnitude indicate excess generated mass on one side of the
cutoff. Aggregate the directional evidence into:

- a per-particle surplus score \(s_i\);
- a per-target planning-point deficit score \(d_j\);
- a proposed transported mass fraction \(\rho\); and
- an active direction subset.

The initial \(\rho\) candidates should be predeclared, such as

\[
\rho\in\{0,\;0.05,\;0.10,\;0.20,\;0.40,\;1.0\}.
\]

Select \(\rho\) using only controller-pool diagnostics. Prefer the smallest
near-optimal admissible intervention. This is a trust-region principle: do not
move mass that the local learner already placed correctly.

### 4.4 Expected sliced transport plan

For every active direction \(u_\ell\), sort generated and planning-target
projections and construct the one-dimensional rank-matching permutation
\(\pi_\ell\). Lift it to a permutation coupling:

\[
\Gamma_\ell = \frac{1}{N}P_{\pi_\ell}.
\]

Average the lifted plans:

\[
\Gamma_{\mathrm{EST}}
=
\frac{1}{L}\sum_{\ell=1}^{L}\Gamma_\ell.
\]

Because each \(\Gamma_\ell\) has the correct empirical marginals, their average
is a valid doubly stochastic coupling. This is a coherent joint object,
unlike independently averaging scalar projected destinations.

The simplest barycentric destination is

\[
T_i =
N\sum_j\Gamma_{\mathrm{EST},ij}y_j
=
\frac{1}{L}\sum_\ell y_{\pi_\ell(i)}.
\]

This arm is required as a diagnostic, but it is not assumed to be sufficient:
a barycenter can still average target points from disconnected modes.

### 4.5 Consensus or partial joint plan

Use \(\Gamma_{\mathrm{EST}}\) to construct a sparse candidate graph rather than
immediately taking its barycenter. A starting cost is

\[
C_{ij}
=
\alpha_x\|x_i-y_j\|^2
-\lambda_{\mathrm{slice}}\log
  \left(\Gamma_{\mathrm{EST},ij}+\varepsilon\right)
+\lambda_{\mathrm{support}}C^{\mathrm{support}}_{ij}.
\]

The support term may use multiscale target-neighbor geometry or local tangent
information, but it must be label-free and calibrated only on target samples.

Solve a sparse entropic partial-transport problem:

\[
\min_{\gamma\geq 0}
\langle C,\gamma\rangle
+\varepsilon_{\mathrm{OT}}
  \operatorname{KL}(\gamma\|a\otimes b)
\]

subject to

\[
\gamma\mathbf 1\leq a,\qquad
\gamma^\top\mathbf 1\leq b,\qquad
\mathbf 1^\top\gamma\mathbf 1=\rho .
\]

The source capacity \(a\) is derived from surplus scores and the target
capacity \(b\) from deficit scores. The planner therefore transports only the
mass selected by the controller.

Required plan variants:

1. EST barycenter;
2. hard consensus assignment from the EST graph;
3. sparse full-mass Sinkhorn on the EST graph;
4. sparse partial Sinkhorn on the EST graph;
5. dense full Sinkhorn reference.

### 4.6 Geometry guard and backtracking

For each proposed transported particle \(z_i(\eta)=x_i+\eta\delta_i\), test
step sizes

\[
\eta\in\{1,\tfrac12,\tfrac14,\tfrac18,0\}.
\]

The controller accepts the largest step that:

1. improves the frozen controller coverage objective by a declared tolerance;
2. does not worsen support precision/leakage beyond a declared tolerance; and
3. remains finite and within a target-calibrated outer scale bound.

The generic support score should be calibrated by target-versus-target
nearest-neighbor distances on the controller pool. Synthetic labels may be
used for evaluation, never for accepting a step.

This guard is not allowed to choose a step from the evaluation pool.

### 4.7 Neural amortization

Neural training begins only after a particle teacher passes Stage 1.

Required neural arms:

1. **single-head endpoint MSE:** the current reference;
2. **persistent-cohort single head:** the same latent cohort keeps its route
   assignment across a macro rollout;
3. **route-conditioned head:** a discrete route identifier is supplied to the
   transport network;
4. **mixture-of-experts generator:** separate experts model disconnected route
   families, with learned but audited mixture weights.

Route identifiers must be constructed without synthetic component labels.
Possible constructions include:

- connected components of a sparse target-neighbor graph;
- clustering of transport destinations;
- stable partitions of the EST/partial-plan support.

The default distillation objective should preserve the coupling rather than
only regress to its barycenter:

\[
\mathcal L_{\mathrm{plan}}
=
\sum_{ij}\gamma_{ij}
\|G_\theta(z_i,c_i)-y_j\|^2.
\]

The full neural loss is

\[
\mathcal L
=
\mathcal L_{\mathrm{plan}}
+\lambda_{\mathrm{anchor}}\mathcal L_{\mathrm{anchor}}
+\lambda_{\mathrm{semantic}}\mathcal L_{\mathrm{semantic}}
+\lambda_{\mathrm{support}}\mathcal L_{\mathrm{support}}.
\]

All terms must be nonnegative and separately reported. Do not argue that a
sum of vector fields is identifiable: cancellation between vector fields can
make the sum zero without making either branch zero. If an exact zero-loss
claim is later made, it must use nonnegative losses and positive weights so
zero total loss forces the anchor loss itself to vanish.

### 4.8 Fixed anchor and semantic branch

The fixed anchor begins with one of:

1. raw data coordinates for low-dimensional experiments;
2. an injective multiscale linear transform with all required scales retained;
3. an explicitly verified injective embedding on the declared data class.

The anchor discrepancy should be mathematically measure-determining in the
ideal population setting, such as the certified Laplace population field or a
characteristic-kernel discrepancy.

The semantic branch may use a pretrained or learned encoder. It is responsible
for perceptual efficiency, not the only correctness guarantee.

An orthonormal wavelet or DCT transform alone merely re-expresses Euclidean
distance. To change optimization behavior while retaining injectivity, use
declared nonzero multiscale weights, multiple bandwidths, or localized
cross-scale comparisons. The unweighted transform is still a useful
implementation control.

## 5. Baselines and fairness

### 5.1 Required baselines

Every relevant confirmation must include:

1. the official-style paper protocol;
2. the frozen current PSQT flagship;
3. the local geometry learner without coverage repair;
4. full dense Sinkhorn transport at matched particle count;
5. the proposed coherent partial-transport candidate.

Where feasible, add a faithful W-Flow or Sinkhorn-Drifting comparator. If the
implementation is not faithful, label it as a local approximation and do not
attribute its performance to the published method.

### 5.2 Two comparison regimes

Report both:

1. **budget matched:** approximately equal target examples, generated
   examples, and declared compute budget;
2. **quality frontier:** cost required to reach fixed ED2, SW1, coverage, and
   support-precision thresholds.

Equal kernel-pair counts do not imply equal FLOPs. The ledger must separately
record:

- target examples read;
- generated examples;
- projections;
- sorting work;
- candidate graph edges;
- Sinkhorn iterations;
- kernel evaluations;
- optimizer updates;
- wall time;
- peak CPU/GPU memory;
- serialized atlas and planner state.

Atlas construction should be reported both as one-time setup and amortized
over \(1,2,4,\) and \(8\) generators trained against the same target.

### 5.3 Hyperparameter separation

Use three registries:

1. development;
2. frozen validation/model selection;
3. fresh confirmation.

Do not select:

- bandwidth;
- transported fraction;
- active-direction count;
- number of Sinkhorn iterations;
- support radius;
- route count;
- neural capacity;
- checkpoint;

from the confirmation results.

## 6. Target suite

### 6.1 Structured geometry targets

Required two-dimensional targets:

- official noisy checkerboard;
- Swiss roll;
- pinwheel;
- two moons;
- concentric rings;
- separated narrow modes;
- connected filaments with sharp curvature;
- target with a hole or annulus.

These are not isolated showcase examples. They probe different failure modes:
disconnected support, curved support, holes, narrow components, and local
manifold geometry.

### 6.2 Allocation and tail targets

Required targets:

- balanced Gaussian mixtures;
- rare-component mixtures with weights
  \(0.5\%,1\%,2\%,5\%,10\%\);
- anisotropic mixtures;
- correlated mixtures;
- unequal-scale mixtures;
- broad-to-concentrated and concentrated-to-broad initializations.

### 6.3 Dimension and scale stress

Use dimensions:

\[
d\in\{2,4,8,16\}
\]

for the first confirmation and

\[
d\in\{32,64,128\}
\]

only after the low-dimensional gates pass.

Use target scales:

\[
s\in\{0.5,1,2,4\}.
\]

The image/encoder phase is deferred until both the particle and neural
retention gates pass.

## 7. Metric panel

No candidate may be selected using ED2 and SW1 alone.

### 7.1 Global distribution metrics

- energy distance or ED2;
- held-out sliced \(W_1\);
- mean error;
- covariance Frobenius error;
- optional MMD with a preregistered multiscale kernel family.

### 7.2 Support quality

- target-calibrated support precision;
- target-calibrated support recall;
- density and coverage;
- off-support occupancy;
- checkerboard cell leakage and cell-mass \(L_1\);
- manifold or graph distance for curved supports;
- hole-filling error for rings/annuli.

The support threshold must be calibrated from an independent
target-versus-target split.

### 7.3 Rare-mode metrics

- component-core coverage;
- component-core precision;
- rare-component mass error;
- conditional covariance error inside the rare component;
- teacher-to-student rare-core retention.

Nearest-center occupancy is insufficient because bridge particles may be
assigned to the nearest rare mode without lying inside its genuine core.

### 7.4 Neural retention metrics

For the particle teacher and neural student, report:

- endpoint ED2/SW1 ratio;
- support-precision retention;
- support-recall retention;
- rare-core mass retained;
- per-cohort destination variance;
- route-switch frequency;
- disagreement between teacher plan and student endpoints.

### 7.5 Encoder-robustness metrics

For every encoder condition, report:

- source-space metrics;
- feature-space metrics;
- degradation relative to the strong semantic encoder;
- explicit collision-target behavior.

A model is encoder robust only if source-space quality remains acceptable when
the semantic feature is weakened. Matching the damaged feature distribution
is not evidence of robustness.

## 8. Statistical protocol

For development:

- use enough seeds to reject obviously unstable mechanisms cheaply;
- report every registered cell, not only averages.

For confirmation:

- use at least two fresh target realizations per family/dimension cell;
- use at least two initializations where applicable;
- use at least ten optimization seeds per cell, preferably twenty for the
  final low-dimensional claim;
- predeclare the aggregation rule;
- report paired bootstrap confidence intervals for ratios and differences;
- report target-level win counts as secondary diagnostics;
- correct or clearly label families of simultaneous hypothesis tests.

The experimental unit for a broad generalization claim is the target
realization, not every generated particle.

## 9. Implementation stages

### Stage 0: honesty and reproduction harness

Before implementing a new planner:

1. freeze the official-style paper baseline;
2. freeze the current flagship;
3. reproduce checkerboard and Swiss-roll outcomes from clean commands;
4. add the full support metric panel;
5. add exact target-access and compute ledgers;
6. hash registries, outputs, and figure-generating inputs;
7. separate the earlier smooth synthetic registry from the structured
   geometry registry in all documentation.

**Gate 0:** reproduced figures and metrics must agree with the consumed
artifacts within a declared numerical tolerance.

### Stage 1A: particle-level coherent-plan screen

Implement the EST coupling and compare:

1. paper drift;
2. current independent PSQT correction;
3. EST barycenter;
4. hard EST consensus;
5. sparse full-mass EST/Sinkhorn;
6. dense Sinkhorn reference.

Use free particles only. Do not train a neural network.

**Mechanism outputs:**

- coupling marginal error;
- number of target candidates per source;
- plan entropy;
- bridge/leakage change;
- rare-mass change;
- ED2/SW1;
- time and memory.

**Gate 1A:** at least one coherent-plan arm must improve support precision or
leakage over current PSQT while retaining most of its global and rare-mode
gain. The exact non-inferiority margins must be frozen before the run.

If no arm passes, stop neural PSQT development and retain the KLL atlas only as
an audit/diagnostic tool.

### Stage 1B: partial-transport controller

Hold the best Stage 1A planner fixed. Implement:

- surplus/deficit scores;
- candidate transported fractions;
- controller-pool selection;
- geometry guard;
- backtracking;
- local-first and interleaved scheduling.

Factorial comparison:

| Planner | Transported mass |
|---|---|
| current PSQT | full |
| coherent planner | full |
| coherent planner | fixed partial |
| coherent planner | adaptive partial |

**Gate 1B:** adaptive partial transport must improve the declared combined
coverage/precision endpoint over full transport and must not merely select
\(\rho=1\) everywhere.

If it selects \(\rho=0\) almost everywhere, simplify the model to the local
learner and report that persistent repair was unnecessary in this scope.

### Stage 2: fresh particle confirmation

Freeze:

- planner;
- sparse graph rule;
- controller;
- schedule;
- support guard;
- all thresholds;
- target suite;
- seeds and statistical analysis.

Run fresh targets and compare with paper, current PSQT, and dense Sinkhorn.

**Promotion gate:** the candidate must:

1. retain PSQT's rare-mode advantage;
2. be non-inferior to paper support precision/leakage;
3. improve at least one preregistered global endpoint over paper;
4. improve the primary combined endpoint over current PSQT;
5. report all additional cost.

Only a promoted particle teacher proceeds to neural amortization.

### Stage 3A: persistent-cohort neural baseline

Implement persistent target-plan assignments for a fixed latent cohort.
Compare with the existing fresh-reranking MSE student under matched capacity
and optimizer budget.

**Gate 3A:** persistent cohorts must improve teacher-to-student retention,
especially rare-core mass and support precision.

### Stage 3B: route-aware neural model

Implement route-conditioned and mixture-of-experts arms. Compare:

1. current single head;
2. persistent-cohort single head;
3. route-conditioned shared trunk;
4. mixture-of-experts with matched parameter count;
5. mixture-of-experts with matched wall budget.

Report route occupancy and collapse. Reject any route model whose improvement
comes only from materially greater capacity without a matched-capacity win.

**Gate 3B:** the selected student should retain at least a preregistered large
fraction of the teacher's rare-core and support advantage. A recommended
starting requirement is \(80\%\), compared with the roughly \(44\%\) retention
seen in the earlier failure analysis.

### Stage 4: encoder-dependence study

Hold the promoted architecture fixed and compare:

1. feature-only strong semantic encoder;
2. feature-only weak bottleneck;
3. feature-only frozen random encoder;
4. feature-only intentionally lossy encoder;
5. fixed anchor only;
6. fixed anchor plus strong semantic encoder;
7. fixed anchor plus each damaged semantic encoder.

Include explicit collision targets that the lossy encoder cannot distinguish.

**Gate 4:** anchor-plus-semantic must:

- match or nearly match the strong feature model under the good encoder;
- degrade less under weak/lossy encoders;
- avoid the collision failure in source-space metrics;
- report its extra cost.

If the fixed anchor helps correctness but is too weak to drive optimization,
retain it as a guard or certification loss rather than claiming it replaces
semantic features.

### Stage 5: cost reduction

Only after quality gates pass:

1. reuse active direction subsets;
2. compare exact quantiles and KLL summaries;
3. prune EST edges;
4. warm-start partial Sinkhorn;
5. reduce Sinkhorn depth;
6. test weighted target coresets;
7. benchmark cached attraction summaries where mathematically compatible.

Produce quality-cost Pareto curves rather than one selected point.

**Gate 5:** promote an optimization only if its confidence interval remains
inside the frozen quality non-inferiority margins.

### Stage 6: higher-dimensional and image-feature bridge

Proceed to dimensions 32--128, then small real-feature tasks. Do not start
with ImageNet.

Required intermediate datasets should have:

- known or auditable support structure;
- more complexity than synthetic mixtures;
- manageable exact or high-quality reference metrics;
- several encoder conditions.

Only after this bridge succeeds should an image-generation collaboration or
large-scale benchmark be proposed.

### Stage 7: formal follow-up

Formalization follows mechanism validation rather than preceding it.

Potential formal targets:

1. the EST average has the correct empirical marginals;
2. partial-plan capacities preserve total transported mass;
3. a nonnegative anchor loss prevents feature-only source collisions at exact
   zero;
4. the ideal fixed anchor identifies the source distribution;
5. a quantitative anchor stability certificate transfers small anchor loss to
   a stated source discrepancy.

Do not attempt to prove that zero of a sum of vector fields makes each
component zero. Use separately nonnegative losses or an explicit lexicographic
constraint.

## 10. Software architecture

Keep the new work isolated from consumed results.

Recommended modules:

```text
numerics/coherent_transport/
    atlas.py
    est_plan.py
    sparse_partial_ot.py
    coverage_controller.py
    support_guard.py
    routing.py
    anchor_features.py
    metrics.py
    cost_ledger.py
    registries.py
```

Recommended experiment entry points:

```text
run_coherent_particle_screen.py
run_partial_controller_development.py
run_coherent_particle_confirmation.py
run_route_amortization_development.py
run_encoder_robustness_study.py
analyze_coherent_transport.py
plot_coherent_transport.py
```

Every run directory should contain:

- source commit/hash;
- full configuration;
- registry hash;
- environment information;
- stdout/stderr;
- raw per-target/per-seed metrics;
- aggregate analysis;
- cost ledger;
- model/checkpoint hash where applicable.

Do not overwrite the current confirmed PSQT artifacts.

## 11. Required unit and property tests

### EST plan

1. every rank plan is a permutation coupling;
2. row and column masses equal \(1/N\);
3. the average plan preserves both marginals;
4. identical source and target samples admit the identity plan under stable
   tie-breaking;
5. results are deterministic under the frozen tie rule;
6. target sample identities are retained correctly.

### Sparse/partial plan

1. nonnegative coupling;
2. source and target capacities are respected;
3. transported mass equals \(\rho\) within tolerance;
4. \(\rho=0\) is exactly no repair;
5. \(\rho=1\) agrees with the declared full-mass variant;
6. dense candidate graph agrees with dense reference within tolerance;
7. no NaN/Inf at extreme scales;
8. gradients are finite where used neurally.

### Controller and guard

1. evaluation data and directions are inaccessible;
2. exact match proposes zero or negligible intervention;
3. a known missing-mode case proposes positive intervention;
4. adversarial off-support moves are backtracked or rejected;
5. selected mass and gains replay deterministically;
6. target-label information is never used.

### Neural routing

1. persistent cohorts retain identifiers across steps;
2. route IDs do not depend on evaluation labels;
3. matched-capacity parameter counts are verified;
4. inactive routes are detected;
5. route collapse and occupancy are reported;
6. inference does not require target labels.

### Anchor branch

1. fixed transform is numerically injective on the declared finite domain;
2. inverse/reconstruction error is reported where applicable;
3. every retained scale has nonzero weight;
4. feature-only collision test fails as expected;
5. anchor branch distinguishes the collision pair;
6. anchor and semantic metrics are logged separately.

## 12. Promotion gates in compact form

| Gate | Required evidence | Failure action |
|---|---|---|
| G0 Reproduction | Official-style baseline and current flagship reproduced | Repair harness before research |
| G1 Coherent plan | Better geometry than current PSQT with retained coverage | Demote PSQT to diagnostics |
| G2 Partial control | Better precision/coverage balance than full repair | Use full plan or remove controller |
| G3 Particle confirmation | Fresh win over current PSQT and fair paper baseline | Do not amortize |
| G4 Neural retention | Large fraction of teacher gain retained | Fix routing/capacity, not projections |
| G5 Encoder robustness | Anchor reduces source-space degradation under encoder damage | Keep encoder claim narrow |
| G6 Cost | Favorable quality-cost frontier | Report quality method without efficiency claim |
| G7 Scale bridge | Stable 32--128D and real-feature results | Do not claim image readiness |

## 13. Failure interpretation

| Outcome | Interpretation | Next move |
|---|---|---|
| EST fixes bridges | Independent slices were the key defect | Build partial controller |
| EST barycenter still bridges, hard plan works | Averaging target identities is the defect | Keep discrete/hard routing |
| Full Sinkhorn wins, sparse plan fails | Candidate graph is too restrictive | Increase graph adaptively or accept dense cost |
| Partial plan beats full plan | PSQT is valuable as an intervention controller | Promote selective repair |
| Particle teacher wins, student fails | Amortization/routing is the bottleneck | Route-aware or multi-generator architecture |
| Persistent cohorts alone fix student | Re-ranking instability was dominant | Avoid unnecessary mixture complexity |
| Mixture routes fix support | Connected single-generator topology was limiting | Promote discrete latent routing |
| Anchor helps collision only | Anchor is a correctness guard, not main optimizer | Keep small positive anchor weight |
| Anchor hurts all quality | Sample complexity/bandwidth is inadequate | Multiscale anchor or narrower claim |
| Better ED2/SW1 but worse precision | Global mass improved at support cost | Candidate fails promotion |
| Better precision but loses rare modes | Repair became too conservative | Revisit selected mass/capacity |
| Cost approaches dense Sinkhorn | Efficiency hypothesis failed | Compare as quality method only |

## 14. Guardrails

1. Do not call the earlier 62/64 result a 64-shape geometry benchmark.
2. Do not select models using ED2/SW1 alone.
3. Do not use the weak `tau=1` repository paper port as the only baseline.
4. Do not use target component labels in the algorithm.
5. Do not let evaluation samples or projections influence controller choices.
6. Do not describe KLL summaries as retaining a joint target coupling.
7. Do not claim a barycentric EST destination is mode preserving without
   testing it.
8. Do not claim that feature-law equality implies source-law equality.
9. Do not claim that an injective transform solves finite-sample
   high-dimensional optimization.
10. Do not hide atlas construction, sorting, or transport iterations from the
    compute ledger.
11. Do not combine several new mechanisms before each passes an isolated
    development gate.
12. Do not proceed to neural or image-scale experiments after a failed
    particle-level gate.
13. Preserve every negative result and consumed artifact.

## 15. Immediate implementation order

The recommended concrete sequence is:

1. create the clean structured-geometry benchmark and reproduce paper/current
   flagship results;
2. implement target-calibrated support precision, recall, leakage, density,
   coverage, and manifold metrics;
3. implement and unit-test the EST plan;
4. run the free-particle EST barycenter and hard-consensus screen;
5. implement sparse full and partial transport on the EST candidate graph;
6. run the Stage 1A comparison against current PSQT and dense Sinkhorn;
7. stop and analyze before building the adaptive controller;
8. implement the selected-mass controller and geometry guard only if a
   coherent plan passes;
9. freeze and run the fresh particle confirmation;
10. begin persistent/discrete neural routing only after the particle teacher
    is promoted;
11. run the encoder-stress experiment after neural retention is established;
12. optimize cost last.

The first decisive deliverable is therefore not a neural model. It is a
particle-level table and visual atlas answering:

> Does converting persistent sliced evidence into a valid, selective joint
> plan preserve target geometry while retaining rare-mode and global-mass
> gains?

## 16. Expected research contribution if successful

A successful result would be narrower and more defensible than claiming that
PSQT universally outperforms the drifting paper.

The intended contribution is:

> Persistent quantile summaries are used as a low-memory coverage controller,
> a coherent partial plan transports only demonstrably deficient mass, a local
> drifting branch preserves geometry, and an explicit source-space anchor
> reduces dependence on the learned semantic encoder.

This differs from full Sinkhorn drifting in that the persistent sliced state
selects and sparsifies the intervention rather than replacing every local
update with a dense global plan. The strongest empirical claim would be a
better coverage/precision/cost frontier under fair structured-geometry and
encoder-stress tests.

## 17. Relevant primary literature

- Original Drifting Models paper:
  <https://arxiv.org/abs/2602.04770>
- Expected Sliced Transport Plans:
  <https://proceedings.iclr.cc/paper_files/paper/2025/hash/98db6567a141db93b3cdeb177da8ab37-Abstract-Conference.html>
- Conditional Sliced-Wasserstein Flows:
  <https://proceedings.mlr.press/v202/du23c.html>
- Random-Path Projected Wasserstein Distance:
  <https://proceedings.mlr.press/v235/nguyen24l.html>
- Partial minibatch optimal transport:
  <https://proceedings.mlr.press/v162/nguyen22e.html>
- Multisample Flow Matching:
  <https://proceedings.mlr.press/v202/pooladian23a.html>
- W-Flow:
  <https://arxiv.org/html/2605.11755>
- Sinkhorn-Drifting:
  <https://arxiv.org/abs/2603.12366>
- DriftXpress:
  <https://arxiv.org/abs/2605.12183>
- Kernel-Gradient Drifting:
  <https://arxiv.org/html/2605.10727>
- Continuous-generator disconnected-support limitation:
  <https://proceedings.mlr.press/v119/tanielian20a.html>
- Improved precision and recall:
  <https://proceedings.neurips.cc/paper/2019/hash/0234c510bc6d908b28c70ff313743079-Abstract.html>
- Density and coverage:
  <https://proceedings.mlr.press/v119/naeem20a.html>

## 18. Recommended first checkpoint

Stop after Stage 1A and produce:

1. tested EST and partial-plan primitives;
2. a reproducible free-particle geometry benchmark;
3. visual target/paper/current-PSQT/EST/partial-OT comparisons;
4. the complete metric and cost panel;
5. a written go/no-go decision for neural amortization.

That checkpoint is intentionally early. If the joint planner does not repair
geometry before amortization, no amount of neural tuning should be expected to
make the mechanism reliable.
