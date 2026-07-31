# Quantile-Guarded Drifting: Research Conclusion and Implementation Plan

**Status:** proposed successor to LB-QCD; not yet implemented or empirically
validated  
**Scope:** low-dimensional learned-generator experiments, initially the existing
one-dimensional missing/concentrated-mode benchmark family  
**Primary objective:** produce a method that improves materially and
statistically reliably over LB-QCD, the repository's strongest current
learned-generator method—not merely over a globally tuned paper baseline  
**Date of research pass:** 2026-07-21

## 1. Executive decision

The next implementation should not be another kernel, mask, scalar gain,
bandwidth selector, occupancy router, or hard phase schedule. The repository
already contains enough negative evidence against those directions.

The strongest remaining hypothesis is:

> Keep LB-QCD's global rank-transport phase, but replace the hard transition to
> an unconstrained paper-field suffix with a simultaneous optimizer-space
> update. The local drifting proposal is accepted or minimally corrected so
> that it cannot erase a prescribed fraction of the rank objective's local
> progress. Select the final checkpoint using independent validation samples,
> because the local fields repeatedly showed useful transient states followed
> by endpoint erosion.

This document calls the proposed method **Quantile-Guarded Drifting (QGD)**.
The primary version uses the original paper Laplace field as its local signal,
because that field has been the best late-training stabilizer in this codebase.
A sharp/log-KDE version is a secondary ablation, not the primary candidate.

The design deliberately combines the three most reliable empirical facts in
the repository:

1. exact rank transport gives the largest improvement that transfers through a
   learned generator;
2. the paper field remains useful for local stabilization;
3. sequential local refinement can improve an intermediate state and then
   erase part of the gain before the fixed endpoint.

There is no guarantee that QGD will beat LB-QCD before it is tested. It is,
however, the first proposed successor that attacks the observed failure at the
layer where it occurs: the interaction of two useful update signals after they
pass through the generator Jacobian and Adam.

## 2. Repository evidence

Ratios below are lower-is-better. Results from different frozen campaigns are
not a strict common leaderboard; within-campaign comparisons are the reliable
unit of evidence.

| Attempt | Primary result | What it established |
|---|---:|---|
| Geometry/mask/step modifications | aggregate ED2/paper about `1.031` | no general benefit from the first heuristic policy |
| Conditional mask policy | fresh held-out aggregate ratio `1.000` | useful on some curved non-Gaussian targets, but the router did not generalize |
| NCJ particle dynamics | ED2/paper `0.100`, CI `[0.050, 0.195]` | removing the `P*Q` gain and cross-fitting can transform particle dynamics |
| NCJ learned generator | ED2/paper `1.072`, CI `[0.946, 1.208]` | the particle win did not survive generator parameterization and Adam |
| QLD-v1 | ED2/paper `0.9105`, CI `[0.8270, 0.9873]` | exact rank transport plus a paper suffix is a real trained-generator improvement |
| QLD-v1 | kernel-pair ratio `0.300`, wall ratio about `0.498` | the improvement was also computationally attractive |
| LB-QCD | ED2/paper `0.8218`, CI `[0.7649, 0.8915]` | strongest current trained-generator result |
| LB-QCD | ED2/paper-oracle `0.9437` | its benefit is not merely failure to tune the paper bandwidth per target |
| LB-QCD | ED2/QLD about `0.983` | RSR resolution gating adds a small but useful increment; QLD supplies most of the gain |
| OA-SQD | ED2/QLD `1.0031` | better sampling/controller machinery did not improve the endpoint |
| Sharp conservative suffix | best tested endpoint/QLD about `1.015` | no fixed-endpoint win |
| Sharp conservative trajectory | hindsight best/handoff `0.5206`, endpoint/handoff `0.7864` | a potentially valuable transient signal followed by erosion; hindsight is not a valid selector |
| Calibrated bridge | best ED2/QLD `0.9842`, SW1/QLD `0.9997` | small signal, but no joint ED2/SW1 win |
| Full QLD continuation | ED2/QLD-v1 `1.1835` | the paper suffix is not dispensable at the tested horizon |

The supporting repository reports are:

- [LowDimAttributionResults.md](LowDimAttributionResults.md)
- [IdentifiabilityImprovementResults.md](IdentifiabilityImprovementResults.md)
- [QuantileFissionConfirmatoryResults.md](QuantileFissionConfirmatoryResults.md)
- [LBQCDConfirmatoryResults.md](LBQCDConfirmatoryResults.md)
- [OASQDDevelopmentResults.md](OASQDDevelopmentResults.md)
- [ConservativeFinisherResults.md](ConservativeFinisherResults.md)
- [CalibratedBridgeResults.md](CalibratedBridgeResults.md)

### 2.1 What worked

#### Exact one-dimensional rank transport

For equal-size empirical clouds, sorting both clouds and pairing equal ranks
gives the exact empirical one-dimensional quadratic optimal transport map. In
the implementation,

\[
v_Q(x_i)=y_{\operatorname{rank}(x_i)}-x_i.
\]

The resulting stop-gradient generator update was the first modification to
produce a reliable learned-generator gain. It is especially effective at
global allocation: relocating generated probability mass between modes,
matching quantiles, and avoiding the locality limitations of a narrow kernel.

#### Resolution-gated virtual batches

LB-QCD improved QLD modestly by using Run-Sort-ReRun only when the ordinary
training batch could not resolve significant quantile regions. This mechanism
is useful but should remain frozen initially: it added about 1.7% to the
confirmatory aggregate while increasing generator-example evaluations by about
6.6 times and wall time by about 12% relative to paper.

#### The paper field as a stabilizer

The full-QLD arm deteriorated at the long horizon, while the 70% QLD / 30%
paper schedule remained strong. The paper field therefore supplies something
that rank matching alone does not: stable local adjustment under the generator
and optimizer used in the benchmark.

#### Short conservative refinement

Sharp and kernel-gradient fields often improved the handoff substantially
before their endpoints deteriorated. This is not evidence that the hindsight
minimum is attainable, but it is evidence that fixed phase duration is
discarding potentially useful states.

### 2.2 What failed and should not be the next primary direction

- **Bandwidth selection:** both field-cosine and cloned-Adam lookahead
  selectors lost to fixed `tau = 0.5`.
- **Positive gain engineering:** Adam largely removed the intended global
  magnitude effects. Tempered `(P*Q)^gamma` schedules were nearly flat.
- **Cross-fitting and jitter for generators:** the large particle benefit did
  not transfer; cross-fitting also increased generator work.
- **Occupancy routing:** the estimator and work accounting were sound, but the
  controller tied QLD rather than improving it.
- **More RSR everywhere:** full RSR tied ordinary QLD. Resolution estimation
  was not the endpoint bottleneck once the simple hybrid was present.
- **Pure conservative suffixes:** stable and mathematically attractive, but no
  tested fixed endpoint beat QLD-v1.
- **Hard bridges:** optimizer calibration, state copying, reset variants, and
  trust caps did not prevent erasure strongly enough.
- **Pure QLD continuation:** it removed the late stabilizer and lost badly at
  the full horizon.

The bridge's deleted-negative ESS was healthy (minimum about `4.60`, median of
per-row minima about `13.30`). Denominator collapse is therefore not the
leading explanation for the present endpoint gap.

## 3. Literature synthesis

The proposal is a synthesis of established ideas, specialized to the exact
failure exposed by this repository.

### 3.1 Mixed distributional flows are legitimate

[Gradient Flow Drifting](https://arxiv.org/html/2603.10592) identifies Gaussian
drifting with a KDE-smoothed forward-KL Wasserstein gradient flow. Its Theorem
4.12 establishes that a positive sum of identifiable divergences remains a
valid divergence and that its Wasserstein velocity is the corresponding sum of
velocities. The paper motivates mixed flows because different divergences have
complementary precision and coverage behavior.

This supports using more than one distributional objective. It does **not** by
itself solve the finite-generator optimization problem observed here.

### 3.2 Distributional gradients can be aggregated as a multi-objective problem

[Multiple Wasserstein Gradient Descent](https://arxiv.org/abs/2505.18765)
estimates separate Wasserstein gradients for several distributional objectives
and combines them through a min-norm/common-improvement calculation. This is
the closest published template for treating QLD and local drifting as
simultaneously useful distributional signals rather than mutually exclusive
phases.

Plain min-norm aggregation is not sufficient for this project: when objectives
strongly oppose, it can choose a very small direction. QGD therefore gives the
rank objective an explicit progress floor.

### 3.3 Gradient conflict should be handled after parameterization

[PCGrad](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)
projects away interfering gradient components. [CAGrad](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)
balances average progress against the worst local objective improvement.

The NCJ particle/generator reversal makes these methods especially relevant:
the useful geometry must be checked after multiplication by the generator
Jacobian and after optimizer preconditioning. A particle-field cosine cannot
certify the behavior of the actual parameter update.

### 3.4 The existing global and local components are literature-supported

- [Run-Sort-ReRun](https://proceedings.mlr.press/v139/lezama21a.html)
  supports virtual large-batch sliced-Wasserstein/rank training when empirical
  resolution is the bottleneck.
- [Drifting Fields Are Not Conservative](https://arxiv.org/html/2604.06333)
  derives the sharp/log-KDE scalar objective and shows that it can be an
  effective conservative alternative. It also makes clear that sharp and
  original drifting generally have different directions.
- [Sinkhorn-Drifting](https://arxiv.org/html/2603.12366) shows that global
  marginal balancing improves low-temperature mode coverage, but at greater
  training cost. It is a useful comparator, not the most direct next step in
  the current one-dimensional suite.
- [Kernel-Gradient Drifting](https://arxiv.org/html/2605.10727) reports gains on
  controlled geometry-sensitive tasks. The repository's own suffix screen did
  not show a general improvement over QLD, so it remains an ablation.

No exact protected combination of LB-QCD/rank transport and paper drifting in
generator-optimizer space was found in this research pass. Hybrid objectives
and multi-objective gradient methods are not new, so novelty must not be
claimed without a broader dedicated review.

## 4. Proposed algorithm

### 4.1 Notation

Let `G_theta` be the generator, `z_i` a latent batch,

\[
x_i=G_\theta(z_i),
\]

and `y_i` an independent target batch of equal size.

At every guarded local update, compute two stop-gradient fields on the same
generated cloud:

1. rank/quantile displacement `v_Q`;
2. local drifting displacement `v_D`.

Their parameter pseudo-gradients are

\[
g_Q=-\frac1N J_\theta^\top v_Q,
\qquad
g_D=-\frac1N J_\theta^\top v_D.
\]

For the original paper field, `g_D` is the gradient of the current frozen
stop-gradient surrogate—not the gradient of one fixed global scalar
functional. Any theorem or documentation must preserve that distinction. The
sharp/log-KDE ablation does possess a scalar local objective.

### 4.2 Phase A: unchanged LB-QCD prefix

Use the frozen LB-QCD policy for the first 70% of updates:

- ordinary direct QLD at the standard batch size when resolution is adequate;
- the existing `M = 1024` RSR update only when the frozen resolution gate
  activates.

Do not tune the gate, virtual batch, training fraction, or paper bandwidth in
the first QGD campaign. Changing them would destroy causal attribution.

### 4.3 Phase B: dual optimizer proposals

Replace the ordinary paper suffix with guarded local updates.

Maintain two independent Adam states:

- `state_Q`, updated only with `g_Q`;
- `state_D`, updated only with `g_D`.

At the LB-QCD handoff, initialize both from exact copies of the carried LB-QCD
Adam state. This preserves the historical paper suffix's carried-state
semantics in `state_D` and avoids giving either objective an artificial reset
advantage. The copies must be independent after initialization. A reset-state
variant is not part of the primary screen because the calibrated-bridge
campaign already found that resets were not the missing mechanism.

Both states observe the same current shared parameter vector. Simulate one
proposal from each state without modifying the shared parameters:

\[
\Delta_Q=\operatorname{AdamProposal}(g_Q,state_Q),
\qquad
\Delta_D=\operatorname{AdamProposal}(g_D,state_D).
\]

After both proposals are computed, advance the two objective-specific moment
states and apply exactly one shared parameter displacement chosen by the
projection below. No third Adam state is needed.

This construction avoids pretending that raw gradient magnitudes are
meaningful under Adam. It also prevents momentum accumulated for one objective
from silently becoming the moment state of the other objective.

### 4.4 Quantile-protected projection

Let `M` be a positive diagonal metric. The primary choice is the diagonal
preconditioner derived from the quantile Adam state, with a numerical floor.
The identity metric must be retained as an ablation.

Define the required rank progress

\[
c_Q=\rho\min(0,g_Q^\top\Delta_Q),
\]

where `rho` is initially one of `0.05`, `0.10`, or `0.20`. Define the local
non-ascent threshold `c_D = 0`.

Choose

\[
\begin{aligned}
\Delta^*=\arg\min_\Delta\quad&
\frac12(\Delta-\Delta_D)^\top M^{-1}(\Delta-\Delta_D)\\
\text{subject to}\quad&g_Q^\top\Delta\le c_Q,\\
&g_D^\top\Delta\le 0.
\end{aligned}
\]

Interpretation:

- if the drifting proposal already makes enough rank progress, apply it
  unchanged;
- otherwise, add the smallest correction in the Adam metric that protects the
  rank objective;
- require the result not to ascend the current local surrogate when the two
  requirements are compatible.

This is a first-order safeguard only. It is not a finite-step monotonicity
theorem, so a trust-radius cap and post-update diagnostics remain necessary.

### 4.5 Closed-form active-set solver

The quadratic program has only two linear constraints. It must be solved by an
auditable active-set enumeration, not a general black-box solver.

Write

\[
a=g_Q,\quad b=g_D,\quad \Delta_0=\Delta_D,
\]

and define

\[
A=\begin{bmatrix}a^\top\\b^\top\end{bmatrix},
\qquad
c=\begin{bmatrix}c_Q\\0\end{bmatrix}.
\]

For an active set `I`, the KKT candidate is

\[
\lambda_I=(A_I M A_I^\top)^{-1}(A_I\Delta_0-c_I),
\qquad
\Delta_I=\Delta_0-MA_I^\top\lambda_I.
\]

Check, in order:

1. no active constraints: accept `Delta_D` if both inequalities hold;
2. quantile constraint only;
3. local constraint only;
4. both constraints.

A candidate is valid only if all multipliers are nonnegative within tolerance
and all primal inequalities hold within tolerance. For the two-active case,
solve the explicit `2 x 2` Gram system. Add only a declared numerical ridge,
and log every ridge or singular fallback.

If no candidate is reliable because the gradients are collinear and opposing,
apply a safe quantile-preconditioned step and record `incompatible=true`. Do
not silently drop the primary rank guard.

### 4.6 Safe proposal fallback

Adam momentum can occasionally produce `g_Q^T Delta_Q >= 0`. In that case the
quantile proposal itself is not a current first-order descent step. Replace it
for purposes of the constraint by

\[
\Delta_Q^{safe}=-\eta_Q M g_Q,
\]

with `eta_Q` calibrated once from the median norm of recent valid quantile
proposals and clipped to a frozen range. Log the fallback frequency. A high
frequency means the dual-state design or handoff initialization is wrong and
must fail the mechanism gate.

### 4.7 Trust radius

Projection can become large when the quantile gradient is tiny or nearly
opposite to the local gradient. Enforce

\[
\|\Delta^*\|_M
\le
c_{trust}\max(\|\Delta_Q\|_M,\|\Delta_D\|_M),
\]

with a predeclared initial `c_trust`, recommended `2.0`. A trust cap is an
upper safety mechanism, not a step-size calibration method. If it activates
often, the candidate should fail rather than be rescued by post-hoc tuning.

## 5. Independent checkpoint selection

The previous conservative experiments observed strong intermediate states but
used the same evaluation trajectories to identify their hindsight minima.
Those minima are diagnostics, not valid algorithm outputs. QGD needs a
predeclared independent selector.

### 5.1 Candidate checkpoints

Save:

- the LB-QCD handoff;
- every tenth local update;
- the final update.

For a 120-step local phase this gives 13–14 candidates, small enough for
reliable independent comparison.

### 5.2 Selection banks

Use target and latent samples disjoint from training batches and from the final
test registry.

- **Bank A:** four paired replicates of at least 4,096 generated and target
  samples per checkpoint. Rank candidates using common random numbers.
- **Bank B:** an independent confirmation bank of at least 8,192 samples,
  applied only to the leading Bank-A candidates.

For finite real datasets, replace resampling by an explicit train / selection-A
/ selection-B / sealed-test split.

### 5.3 Frozen selection score

For checkpoint `t`, normalize both metrics to their handoff values and use

\[
S_t=\frac12\log\frac{ED2_t+\epsilon}{ED2_{handoff}+\epsilon}
   +\frac12\log\frac{SW1_t+\epsilon}{SW1_{handoff}+\epsilon}.
\]

The equal log weights prevent the numerical scale of one metric from
dominating. Coverage and mass error are guards, not extra tunable score terms.

Choose the earliest checkpoint within one standard error of the best Bank-B
score. This biases toward shorter refinement and reduces winner's-curse
selection. The exact epsilon, coverage tolerance, mass tolerance, sample
counts, and tie-break order must be frozen before a standard-scale run.

Do not select separately per target at test time unless the target identity is
legitimately known to the training algorithm and the per-target rule was
predeclared. The primary claim should use one frozen rule.

## 6. Implementation architecture

### 6.1 Proposed new files

- `numerics/quantile_guarded_drifting.py`
  - dual Adam proposal state;
  - gradient flatten/unflatten utilities;
  - two-constraint projection solver;
  - guarded step;
  - checkpoint selector;
  - diagnostics dataclasses and invariant tests.
- `numerics/run_quantile_guarded_development.py`
  - shared handoff construction and branch cloning;
  - staged development screens;
  - work ledger and trajectory output.
- `numerics/QuantileGuardedDriftingProtocol.md`
  - frozen targets, seeds, arms, gates, selection rule, and stopping rules.
- `numerics/qgd_development_registry.json`
  - registry consumed by the runner.
- `numerics/QuantileGuardedDriftingResults.md`
  - generated only after the registered campaign completes.

Use a separate confirmatory runner and registry only after a candidate is
frozen.

### 6.2 Existing components to reuse

From `lbqcd.py`:

- `exact_rank_field`;
- `direct_quantile_step` and RSR logic;
- `paper_step` field semantics;
- `StepWork` and existing compatibility tests.

From the conservative-finisher infrastructure:

- exact model and optimizer snapshots;
- branch-safe handoff cloning;
- separate optimizer-state behavior;
- parameter-gradient and field/loss finite-difference checks;
- trajectory and work-ledger conventions.

From the OA-SQD infrastructure:

- systematic target tables;
- stratified RSR as an optional later cost ablation;
- explicit selection-probability and unbiased-gradient audits.

Do not import private helpers across modules permanently. If QGD needs
`_stopgrad_grads` or Adam simulation from `lbqcd.py`, first promote a small,
tested public interface without changing the historical code path.

### 6.3 Core data structures

Recommended structures:

```text
AdamProposalState
  m, v, step_index

ObjectiveProposal
  gradient
  raw_delta
  metric_norm
  directional_derivative
  next_state

ProjectionDiagnostics
  active_set
  quantile_constraint_before/after
  local_constraint_before/after
  correction_norm
  cosine_gradient
  cosine_delta
  trust_cap_active
  safe_quantile_fallback
  incompatible

CheckpointSelectionRecord
  checkpoint_step
  bank_a_metrics
  bank_b_metrics
  normalized_score
  standard_error
  eligible
```

Every output must be serializable into the run manifest or a dedicated JSONL
diagnostic file.

## 7. Required invariant and regression tests

Before empirical screening, implement the following tests.

### 7.1 Historical compatibility

1. An explicit `guard_enabled = false` compatibility path reproduces the
   unmodified carried-state paper proposal bitwise. Setting `rho = 0` is **not**
   equivalent: it still imposes quantile non-ascent.
2. The LB-QCD prefix is bitwise identical to the frozen implementation for the
   same registry and seeds.
3. Shared handoff clones contain identical parameters, optimizer states, and
   RNG states before branching.
4. Exactly one shared parameter update is applied per training iteration.

### 7.2 Projection algebra

1. Inactive constraints return `Delta_D` exactly.
2. Quantile-only activation matches the one-constraint closed form.
3. Local-only activation matches its closed form.
4. Two active constraints satisfy the KKT system and both equalities.
5. All accepted candidates satisfy primal feasibility within tolerance.
6. Multipliers for active constraints are nonnegative within tolerance.
7. Identity and diagonal metrics agree when the diagonal metric is identity.
8. Singular and exactly opposing gradients trigger the declared fallback.
9. Zero quantile gradient never divides by zero and does not create a false
   progress claim.
10. The trust cap never changes an already admissible update unless the norm
    threshold is actually exceeded.

### 7.3 Gradient and proposal correctness

1. The rank stop-gradient parameter gradient agrees with finite differences of
   its frozen surrogate.
2. The paper gradient remains identical to the historical paper step.
3. The sharp ablation agrees with the certified scalar loss gradient.
4. Simulated Adam proposals match a cloned model taking one real Adam step.
5. Advancing one objective state cannot mutate the other state.
6. Post-update measured directional derivatives match the logged values.

### 7.4 Selection integrity

1. Training RNG streams cannot be read by either selection bank.
2. Bank A and Bank B use independent seeds and samples.
3. Test-registry seeds are rejected if supplied to a development run.
4. The earliest-within-one-SE rule is deterministic under ties.
5. Selecting from a constant metric trajectory returns the earliest eligible
   checkpoint.

### 7.5 Accounting

Record separately:

- generator forward calls;
- generator example evaluations;
- backward example evaluations for each objective;
- target samples;
- kernel pairs;
- sort work;
- selection-only forward evaluations;
- wall time;
- checkpoint-storage cost.

QGD is allowed one rank backward pass and one local backward pass per guarded
local update. Hidden extra lookahead training is not allowed.

## 8. Experimental ladder

### 8.1 Stage Q0 — unit and smoke audit

Run all invariants on tiny deterministic arrays and one short generator smoke
trial. No candidate selection occurs here.

Exit only if:

- every invariant passes;
- all metrics and parameters are finite;
- the baseline compatibility arms are exact;
- work accounting reconciles.

### 8.2 Stage Q1 — mechanism screen

Use development targets only. Previously opened confirmatory targets may be
used for debugging but cannot support a new claim.

Create one shared frozen LB-QCD handoff per target/initialization/seed and clone
it into these arms:

1. historical LB-QCD paper suffix;
2. periodic anchor: four paper steps followed by one direct QLD step;
3. fixed simultaneous mix with a 10% normalized quantile contribution;
4. projected paper QGD, `rho = 0.05`;
5. projected paper QGD, `rho = 0.10`;
6. projected paper QGD, `rho = 0.20`;
7. projected sharp-deleted `tau = 0.2`, `rho = 0.10`.

Use at least three paired seeds. Do not add bandwidths or target-aware routing
at Q1.

The periodic and fixed-mix arms answer whether any persistent rank pressure is
enough. The projected arms answer whether conflict-aware pressure is needed.

### 8.3 Stage Q1 diagnostics

For every arm and cell, report:

- endpoint and independently selected ED2 and SW1;
- exact empirical one-dimensional W2;
- coverage time and censored events;
- mode mass-L1 and worst-mode error;
- handoff, best-selection, and endpoint changes;
- gradient and proposal cosines;
- fraction of updates in each active set;
- median and 90th-percentile projection correction;
- safe-fallback, incompatibility, ridge, and trust-cap rates;
- metrics split by missing versus concentrated initialization;
- metrics split by target family, especially unequal-weight and contaminated
  targets;
- full compute ledger.

Interpretation rules:

- projection never active: the hypothesis reduces to checkpointing or simple
  mixing; do not claim conflict resolution;
- projection active almost always: the local field and rank objective are
  incompatible at the chosen protection level;
- selected gains but endpoint neutrality: the selector is essential and must
  be confirmed independently;
- W2 improves while ED2/SW1 worsen: the guard is too strong or the local field
  is underpowered;
- only missing-mode starts improve: the method is a recovery policy, not yet a
  general successor.

### 8.4 Q1 advancement gate

Advance at most one paper-QGD candidate and optionally one sharp candidate.
The primary candidate should satisfy all of:

- selected ED2/LB-QCD at most `0.95` on the mechanism screen;
- selected SW1/LB-QCD at most `0.98`;
- each predefined family ED2/LB-QCD at most `1.05`;
- each initialization aggregate at most `1.02`;
- no divergence increase;
- safe-proposal fallback below `5%`;
- trust-cap activation below `10%`;
- no accounting or selection-integrity failure.

These are screen gates, not final claims.

### 8.5 Stage Q2 — selection replication

Freeze the candidate and checkpoint-selection rule. Replicate on new
development targets and seeds with:

- endpoint output;
- Bank-A-only selected output;
- Bank-A/Bank-B selected output.

This determines whether the improvement comes from the guarded dynamics, the
selector, or both. Do not proceed if the two-bank advantage disappears.

### 8.6 Stage Q3 — sealed confirmation

Before running, commit:

- candidate code and source hashes;
- all hyperparameters;
- the new target/seed registry;
- the exact paper baseline bandwidth ladder;
- QLD-v1 and LB-QCD comparators;
- metric definitions and bootstrap procedure;
- cost limits and every acceptance criterion.

The same sealed campaign must rerun:

1. validation-selected paper baseline;
2. per-cell hindsight paper bandwidth oracle as a diagnostic only;
3. QLD-v1;
4. LB-QCD;
5. frozen QGD winner.

The final minimum result should be:

- target-balanced ED2/QGD-to-LB-QCD at most `0.95`;
- hierarchical target-bootstrap 95% upper endpoint below `1`;
- SW1/QGD-to-LB-QCD below `1`, preferably at most `0.97`;
- at least 60% target/initialization cell wins;
- every predefined family at most `1.05` relative to LB-QCD;
- no initialization aggregate above `1.02`;
- no worse divergence or censoring;
- all costs fully reported.

If LB-QCD reproduces its earlier approximate `0.8218` ED2/paper ratio, a
genuine `0.95` multiplier would be about `0.781` relative to paper. That would
cross the earlier 20% minimum-effect target. This arithmetic is motivational;
the actual claim must use same-run paired comparisons.

## 9. Failure modes and predeclared responses

### 9.1 Adam proposal is not a descent direction

Use the declared safe preconditioned proposal for that objective and log it. A
high rate fails the candidate; do not tune the fallback after observing test
results.

### 9.2 The two constraints are incompatible

Prioritize quantile progress, apply the safe quantile direction, and log the
event. If incompatibility is common, stop: a hard Pareto conflict has been
found, and this local field is not a viable protected partner.

### 9.3 Projection corrections explode

Activate the frozen trust radius. Frequent activation fails the mechanism
gate. Do not raise the cap merely to rescue a candidate.

### 9.4 Projection is unnecessary

If the projected and fixed-mix arms are equivalent and constraints almost
never activate, keep the simpler method. Complexity must earn a measurable
benefit.

### 9.5 The selector appears to win only on Bank A

Reject the selector. This is ordinary checkpoint winner's curse.

### 9.6 Only one metric improves

Do not call it a general successor. A method that trades ED2 for materially
worse SW1, coverage, or mass calibration must be described as metric-specific.

### 9.7 The method only improves a narrow target family

Record it as a conditionally useful policy. Do not build another target-aware
router until the condition is independently predictable and passes a fresh
held-out gate; the earlier mask and occupancy routers show why this boundary
matters.

## 10. Relationship to the formalization

The formal work informs QGD in three useful but limited ways:

1. the target distribution is a legitimate shared equilibrium for the
   identifying population fields under their stated hypotheses;
2. exact rank/Wasserstein matching is itself identifying in one dimension;
3. positive scaling and zero-set preservation are not enough for learned
   generator performance—conditioning and optimizer interaction matter.

The proposed projection is not yet a Lean theorem, and the current
formalization does not prove that a finite Adam step decreases either
population functional. The QGD implementation must therefore use language
such as **first-order protected update**, not **monotone distributional
descent**, until a theorem with finite-step smoothness assumptions exists.

If the sharp/log-KDE variant succeeds, it offers the cleanest theoretical path:
both rank transport and log-KDE arise from scalar distributional objectives,
so one can later formalize a positive mixed objective or a common-descent
certificate. The original paper-field variant remains the empirically favored
primary arm even though its frozen surrogate changes at each update.

## 11. Claim boundary

Until sealed confirmation, the only valid statement is:

> Quantile-Guarded Drifting is a literature-informed, repository-motivated
> candidate designed to preserve LB-QCD's global mass-allocation progress while
> retaining the paper field's local stabilization.

Even after a successful low-dimensional confirmation, do not claim:

- improvement on ImageNet or real encoder features;
- superiority to the paper's reported FID results;
- a dimension-independent gain;
- convergence of Adam to the population target;
- novelty relative to all hybrid OT or multi-objective generative methods;
- that independent checkpoint selection is free at scale.

A successful Q3 result would support the narrower and still valuable claim:

> On a sealed heterogeneous one-dimensional learned-generator suite,
> quantile-protected local drifting improves both distributional endpoint
> quality and robustness relative to the original paper implementation and the
> repository's previous QLD/LB-QCD methods, under explicit training and
> selection costs.

## 12. Recommended execution order

1. Promote minimal public gradient and Adam-proposal helpers without changing
   historical outputs.
2. Implement and exhaustively test the two-constraint active-set solver.
3. Implement dual objective-specific optimizer states and guarded paper steps.
4. Add the trust radius and full diagnostics.
5. Add independent Bank-A/Bank-B checkpoint selection.
6. Write and commit the QGD development protocol and registry.
7. Run Q0 smoke and invariant audits.
8. Run Q1 mechanism screen with shared handoff clones.
9. Freeze at most one candidate through the registered advancement rule.
10. Run Q2 selection replication on new development targets.
11. If Q2 passes, create and commit a completely new sealed confirmatory
    registry.
12. Run Q3 once, publish every arm and metric, and preserve a failure as a
    valid result.

This ordering first tests the core causal question—whether persistent,
optimizer-aware rank protection prevents local erasure—before spending effort
on larger batches, new kernels, high-dimensional projections, or real-feature
experiments.

## 13. Implementation and execution status (2026-07-21)

Steps 1--8 were implemented. The public Adam/gradient interface, active-set
solver, dual optimizer streams, guarded and fixed-mix steps, trust and fallback
diagnostics, independent two-bank selector, disjoint development registry,
protocol, work ledger, and 14 QGD-specific invariants are present. The Q1
screen completed on all 48 registered target/initialization/seed groups with
no divergence.

The registered advancement gate failed, so steps 9--12 intentionally did not
run. No QGD-v1 arm is frozen for Q2 or Q3. The complete result and diagnosis
are recorded in
[`QuantileGuardedDriftingResults.md`](QuantileGuardedDriftingResults.md), with
machine-readable artifacts under
`qgd_runs/20260721-172034-Q1-screen/`.
