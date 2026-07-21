# Occupancy-adaptive quantile drifting: forensic audit and successor plan

**Date:** 2026-07-21  
**Evidence base:** frozen `LBQCD-confirmatory-v1` result  
**Purpose:** determine what worked in resolution-gated LB-QCD, isolate what
limited it, and specify the next experiment without reusing either frozen
target registry as a tuning set.

## Executive conclusion

Resolution-gated LB-QCD is the first repository candidate with broad,
confirmatory evidence of improvement over the paper implementation in the
scoped one-dimensional learned-generator benchmark. It reduced target-balanced
ED2 by `17.82%` and SW1 by `13.21%`, won `30/32` target/initialization cells,
and remained favorable against a per-cell hindsight paper-bandwidth oracle.
The result is meaningful even though it correctly failed the predeclared
`20%` minimum-effect gate.

The new forensic pass changes the interpretation of the large-batch component:

- the quantile-to-Laplace hybrid supplies most of the aggregate improvement;
- fixed `M=1024` large-batch transport adds only about `1.7%` overall relative
  to QLD-v1;
- nevertheless, in the 18 routed target/initialization cells it reached the
  coverage event earlier in 13 and tied in 5--never later;
- it improved final mass-L1 in 13 of those 18 cells;
- but it improved final ED2 in only 8 of 18 cells.

This is a coherent failure signature: large-batch rank transport is effective
at **early support discovery and mass allocation**, but a target-only router
keeps applying a costly global correction after the generator state no longer
needs it. The fixed 70% warm phase can then damage endpoint geometry that the
ordinary-batch quantile or final paper field would have refined more gently.

The recommended successor is **Occupancy-Adaptive Stratified Quantile
Drifting (OA-SQD)**:

1. construct a tail-aware, bootstrap-stable target quantile atlas from an
   independent target reservoir;
2. monitor generated occupancy of the atlas regions during training;
3. trigger virtual-large-batch rank transport only while a statistically
   supported target region is underrepresented;
4. choose the smallest virtual batch that resolves the active region;
5. form global ranks from that virtual batch, but rerun/backpropagate through
   only a stratified subset with the exact stratum weights;
6. retain the unchanged final paper-only refinement block.

This attacks the measured endpoint and compute defects at the same time. It is
also the cleanest use of the formalization: the atlas regions play the role of
empirical finite basis cells, the controller enforces observable support of
those cells, and positive stratum weights preserve every cell rather than
discarding inconvenient mass.

This document is a research and implementation plan, not a result. The two
existing LB-QCD registries remain sealed and must not be used to choose any
OA-SQD hyperparameter.

## 1. Frozen evidence that must not move

The authoritative result remains
[`LBQCDConfirmatoryResults.md`](LBQCDConfirmatoryResults.md). Its main numbers
are:

| quantity | frozen value |
|---|---:|
| candidate / selected-paper ED2 | `.8218` |
| target-bootstrap 95% interval | `[.7649, .8915]` |
| candidate / paper-oracle ED2 | `.9437` |
| candidate / selected-paper SW1 | `.8679` |
| target/initialization wins | `30/32` |
| worst family ratio | `.8944` |
| candidate / QLD-v1 ED2 | approximately `.9831` |
| summed wall-time / paper | `1.1226` |
| generator evaluations / paper | `6.5945` |
| kernel pairs / paper | `.3000` |
| divergences | `0` |

The correct verdict remains **FAIL**, because `.8218 > .80`. It is a broad,
scoped improvement with insufficient effect size for the frozen promotion
gate--not a failed mechanism and not a claim about ImageNet or higher
dimensions.

## 2. What the implementation actually does

The current candidate has three operational layers:

1. An independent target reservoir of 4,096 samples is split into two halves.
   A spacing diagnostic routes to large-batch transport only if both halves
   contain a sufficiently large and locally cohesive interior gap and an
   ordinary batch of 128 would give the smallest inferred region fewer than
   eight expected samples.
2. A routed run uses Run-Sort-ReRun with `M=1024` for the first 70% of 1,200
   optimizer updates. Every such update evaluates `M` latent samples without
   retained activations, sorts generated and target samples, reruns all `M`
   latents with activations, and takes one full virtual-batch Adam update.
   An unrouted run uses ordinary `B=128` quantile updates.
3. Every run uses the paper Algorithm-2 field at `tau=.5` for the final 30%.

Important positive controls are already present:

- diagnostic and training random-number streams are separate;
- the diagnostic does not receive mixture labels, component counts, true
  weights, or paper outcomes;
- the re-run checks that generator outputs have not changed before the update;
- target and generator work are explicitly counted;
- the confirmatory target registry, protocol, and source snapshots are frozen.

## 3. Strengths revealed by the new pass

### 3.1 The core hybrid generalizes broadly

The result is not supported by one favorable target family. Every predefined
family improved relative to the selected paper baseline, both primary
initializations improved, both ED2 and SW1 improved, and the target-bootstrap
intervals excluded no improvement. The candidate also beat a hindsight paper
bandwidth oracle, so the result is not merely an artifact of comparing against
one poorly selected paper bandwidth.

### 3.2 Global ranks solve an observable support problem

Ordinary `B=128` batches poorly resolve a region with probability mass `p`
when `Bp` is only one or two. A virtual batch gives the monotone coupling a
more stable estimate of the target quantile interval and allows the generator
to receive a correction toward that interval. This is exactly the qualitative
benefit for which Run-Sort-ReRun was designed.

The frozen timing data supports the mechanism rather than merely its endpoint:
large-batch transport was faster or equal on the coverage event in every
routed cell. Its frequent mass-L1 improvement also shows that it reallocates
probability mass effectively even when the final geometric error is worse.

### 3.3 The final paper field is still valuable

The algorithm does not replace drifting. It uses a scale-free global
allocation operator before a local paper-field finishing phase. This division
of labor is consistent with the repository's earlier experiments:

- rank transport is robust to ambiguous local scales and disconnected mass;
- the paper field is useful once support and coarse mass allocation are in
  place;
- ending with the paper field keeps the final equilibrium interpretation close
  to the formally studied model.

### 3.4 The mechanism is simple enough to audit

In one dimension the rank map is exact, deterministic conditional on the
batch, and free of learned critics or component labels. This makes failures
attributable and permits direct gradient-estimator tests. That is a major
advantage over adding a second neural estimator or an opaque clustering
policy.

## 4. Weaknesses and what they imply

### 4.1 Target-only routing cannot know whether the generator needs help

The same target can benefit under one initialization and lose under another:

| routed target | missing-init candidate / QLD | concentrated-init candidate / QLD |
|---|---:|---:|
| C01 unequal, min `.045` | `1.017` | `1.026` |
| C02 unequal, min `.030` | `.862` | `1.044` |
| C03 unequal, min `.020` | `1.021` | `.969` |
| C04 unequal, min `.012` | `1.123` | `1.002` |
| C05 unequal, min `.0075` | `1.035` | `.810` |
| C06 unequal, min `.004` | `.995` | `.904` |
| C10 equal, K24 | `1.046` | `.818` |
| C12 heteroscedastic, K18 | `1.069` | `1.024` |
| C14 contaminated | `.910` | `.862` |

A function of the target alone must make the same conceptual decision in both
states. The table shows that this is the wrong information boundary. Routing
needs a generated-distribution diagnostic.

### 4.2 Fixed duration confuses discovery with refinement

The 13/18 coverage acceleration and 13/18 mass-L1 improvement, combined with
only 8/18 endpoint ED2 improvements, suggest that the large-batch operator is
often useful earlier than it is useful later. The present fixed 70% duration
cannot stop when the rare region has been populated. Continuing full global
rank corrections can trade local shape quality for already-achieved mass
allocation.

The most direct repair is therefore a state-aware stopping rule, not a larger
fixed batch or a longer fixed phase.

### 4.3 The current tail trim hides the rarest nominal component

`diagnose_quantile_resolution` excludes the outer 1% of the empirical
quantile range before considering gaps. Therefore nominal tail components of
mass `.0075` and `.004` cannot be detected as such. C05 and C06 routed because
some other reproducible under-resolved gap was found, not because the
diagnostic established that the named rare tail component was resolved.

This is not data leakage or a protocol violation; it is a semantic mismatch
between the diagnostic's stated purpose and what it can see. Removing the trim
naively would create false positives on connected heavy tails. The successor
needs tail-aware persistence and compactness tests rather than a hard trim.

### 4.4 `M=1024` is simultaneously too large and too small

For a region of mass `.045`, `M=1024` yields about 46 representatives--far
more than needed for a support correction. For mass `.004`, it yields only
about 4.1 representatives and still has roughly a 1.7% chance of seeing none
in an iid target batch. One fixed virtual batch is not an efficient response
to a range of target masses.

The natural rule is

```text
M = smallest allowed power of two with M * p_active >= k_target,
```

with a hard cap and an explicit unresolved state. With `k_target=8`, the
candidate set `{128,256,512,1024,2048}` uses small batches for resolved regions
and reaches about eight expected representatives for a `.004` region.

### 4.5 Full Run-Sort-ReRun spends gradients on ranks that are not scarce

The current RSR update evaluates `M` examples once to construct the global
rank table and a second time to backpropagate through every rank. This is why
generator-example evaluations reach `6.59x` the paper count even though wall
time is only `1.12x` in the small NumPy model.

Only the first pass needs all `M` outputs. Once global correspondences are
known, a properly weighted stratified subset can estimate the full gradient.
This can reduce a global update from `2M` generator evaluations to about
`M+B_back`, retain guaranteed coverage of active rare strata, and restore some
stochasticity.

### 4.6 Full virtual-batch gradients remove potentially useful noise

The current global update is much less noisy than QLD-v1. Reduced rank-boundary
noise is desirable, but eliminating nearly all within-table gradient noise can
make Adam follow a coarse global correction too faithfully. The repository
already contains a generic mean-zero noise-restored RSR primitive, but uniform
subsampling does not guarantee that rare ranks are represented. Stratified
backpropagation is a more targeted version: it preserves global information,
ensures rare-region participation, and remains conditionally unbiased for the
full virtual-table gradient.

### 4.7 The scope remains narrow

The evidence is for one-dimensional synthetic targets, a small tanh generator,
two non-far initialization regimes, and ED2/SW1-style metrics. It does not
establish improvement in multiple dimensions, in real feature spaces, or at
the paper's ImageNet scale. Any successor must first strengthen this scoped
result rather than inflate the claim.

## 5. Literature cross-check

### 5.1 What is directly established

**Virtual global ranks.** Lezama, Chen, and Qiu's
[Run-Sort-ReRun](https://proceedings.mlr.press/v139/lezama21a.html) explicitly
addresses the inability of small sliced-Wasserstein batches to capture target
nuances. It validates the virtual-batch construction, but it does not supply
the state-aware occupancy controller or the proposed rank-stratified backward
pass.

**Minibatch transport is not automatically population transport.**
[Fatras et al.](https://proceedings.mlr.press/v108/fatras20a.html) analyze
minibatch Wasserstein estimators and their gradients, while
[Nguyen et al.](https://proceedings.mlr.press/v162/nguyen22e.html) identify
misspecified minibatch maps and use partial transport to mitigate them. This
supports the diagnosis that increasing batch resolution can improve global
allocation. Partial/unbalanced OT is not the preferred repair here because
the rare and remote target mass is legitimate; allowing it to be discarded
would weaken the exact distribution-matching goal.

**Batch size should depend on estimation quality.** Bollapragada, Byrd, and
Nocedal's
[adaptive sampling framework](https://epubs.siam.org/doi/10.1137/17M1154679)
uses statistical tests to increase sampling effort when a stochastic direction
is not reliable enough. Its exact test is not transplanted here, but it supports
the principle that large batches should be triggered by an observed accuracy
deficit rather than fixed for an entire phase.

**Stratification can lower gradient variance under heterogeneity.**
[SCott](https://proceedings.mlr.press/v139/lu21d.html) provides a concrete
stochastic-optimization example where pre-grouped heterogeneous strata and a
control variate improve the compute/variance tradeoff.
[Adaptive stratified Monte Carlo](https://www.jmlr.org/papers/v16/carpentier15a.html)
formalizes allocation across strata, while
[Carpentier and Munos](https://proceedings.mlr.press/v28/carpentier13.html)
emphasize the tradeoff between finer partitions and reliable allocation. These
results motivate, but do not by themselves prove, the OA-SQD gradient scheme.

**Projection selection matters later.** A 2025
[user's guide to sliced-OT sampling](https://openreview.net/forum?id=ECBepTWAFG)
and 2024
[random-path slicing](https://openreview.net/forum?id=XyxuhLtFA2) document
that direction sampling is a substantive design choice. They are relevant to
a future multidimensional extension, not a reason to mix random projections
into the present one-dimensional mechanism experiment.

**A second drifting-field family now exists.** The 2026
[Gradient Flow Drifting](https://arxiv.org/abs/2603.10592) paper interprets a
Gaussian-kernel drifting field as a KDE forward-KL Wasserstein gradient flow
and proposes mixed reverse-KL/chi-square fields. This is a valuable parallel
experimental branch. It does not directly repair the fixed-duration RSR issue,
and its Gaussian-KDE identities should not be assumed for the practical
L2-Laplace field without a new derivation. Density-ratio weights may also be
least stable in exactly the rare regions studied here.

### 5.2 What the literature does not currently settle

A targeted search through material available on 2026-07-21 did not locate an
exact method combining:

- a target-only persistent one-dimensional quantile partition;
- generated-occupancy confidence tests as a transport controller;
- adaptive virtual batch size based on the active region mass;
- a global Run-Sort-ReRun rank table;
- conditionally unbiased, importance-corrected backpropagation stratified by
  target quantile regions; and
- a final paper-drifting refinement phase.

This is evidence that the combination is a plausible original contribution,
not proof of novelty. A publication-level novelty claim would still require a
formal systematic review and comparison with contemporaneous code and papers.

### 5.3 Ideas deliberately not promoted

| idea | reason it is not first |
|---|---|
| Larger fixed `M` | increases the demonstrated compute defect and does not fix state dependence |
| Longer global phase | contradicts the early-help/late-harm evidence |
| More target-only thresholds | cannot explain opposite outcomes across initializations |
| Unbalanced or partial OT | may improve a metric by deleting legitimate rare mass |
| A generated-sample memory bank | potentially useful, but introduces staleness and estimator-bias questions |
| Mixed KL/chi-square field | scientifically interesting, but orthogonal to the clearest measured defect |
| More bandwidth search | the candidate already beats the hindsight paper oracle; this is no longer the main gap |
| Random sliced extension | previous random-slice results were weak and would obscure one-dimensional attribution |

## 6. Proposed algorithm: OA-SQD

### 6.1 Offline target quantile atlas

Draw an independent reservoir of `N_atlas` target samples, initially 8,192,
and sort it. Construct candidate region boundaries from large adjacent
spacings over the **entire** sample, including the tails.

A boundary is retained only when all of the following hold:

1. both sides have a minimum number of atlas observations;
2. the gap dominates robust local spacing on both sides;
3. the smaller adjacent region is compact relative to the separating gap;
4. its boundary location and estimated mass persist across split halves and
   bootstrap resamples;
5. a connected heavy-tail control does not satisfy the same persistence rule.

Store for every accepted region `j`:

- interval boundaries defined by gap midpoints;
- target mass estimate `p_j`;
- a bootstrap confidence interval for `p_j`;
- local diameter, gap, and persistence diagnostics;
- an explicit `unresolved` flag if the support is too small for a stable
  decision.

The atlas is diagnostic data. Its target-sample cost must be recorded. It may
not use true mixture metadata.

### 6.2 Randomized stratified target table

For a global rank update, do not rely on a fresh iid target sample to happen to
contain a rare region. Build the target table from the empirical atlas using
randomized systematic or region-stratified resampling:

- every selected value is an actual atlas observation;
- each empirical observation has the correct marginal inclusion weight;
- a region receives approximately `M p_j` target ranks;
- randomization changes the selected order statistics between updates.

This separates target-side resolution from generator-side occupancy. The
atlas-size sensitivity must be reported because the algorithm now targets the
empirical atlas distribution conditionally on that reservoir.

### 6.3 Online generated-occupancy controller

Every `H` early-phase updates, draw an independent no-gradient generated probe
of size `N_probe`. Assign generated samples to the atlas intervals and estimate
`q_j`, with simultaneous binomial confidence intervals.

Region `j` is an active deficit only if:

```text
upper_CI(q_j) < lower_CI(p_j) - tolerance
and B_ordinary * p_j < k_target.
```

Use a two-check hysteresis state machine:

```text
LOCAL -> ARMED -> GLOBAL_PULSE -> COOLDOWN -> LOCAL
```

- `LOCAL`: ordinary `B=128` QLD-v1 update.
- `ARMED`: a deficit was seen once; require confirmation at the next check.
- `GLOBAL_PULSE`: perform at most `L_pulse` OA-SQD updates.
- `COOLDOWN`: require two non-deficit checks before another pulse.
- after 70% of training: enter the paper-only phase permanently.

Also leave the global state immediately after two consecutive probes show all
resolved regions within tolerance. The controller responds to the current
generator and can therefore make different decisions for missing and
concentrated starts on the same target.

### 6.4 Adaptive virtual batch size

For the smallest active target mass `p_active`, choose

```text
M = min M_grid such that M * conservative_mass(p_active) >= k_target,
M_grid = {128, 256, 512, 1024, 2048}.
```

Use the lower confidence endpoint as the conservative mass only after checking
the direction of the desired guarantee; otherwise use a direct upper bound on
the probability of insufficient ranks. If no grid value meets the criterion,
log `resolution_cap_hit` instead of silently pretending the region is
resolved.

Initial development constants, not frozen choices:

```text
k_target in {4, 8}
N_probe in {512, 1024}
H in {25, 50}
L_pulse in {10, 25, 50}
```

The grid must be selected on a new development registry.

### 6.5 Rank-stratified backward estimator

The global forward pass produces latent/output/target triples and a monotone
rank displacement `d_i`. Partition rank indices into atlas regions. Let stratum
`j` contain `N_j` virtual ranks and let `g_i` be the stop-gradient per-example
parameter contribution.

The full virtual-table gradient is

```text
g_full = (1/M) * sum_i g_i.
```

Select `b_j >= 1` ranks uniformly without replacement from each active
stratum, allocate remaining backward slots proportionally or by a clipped
displacement-variance proxy, and use

```text
g_hat = sum_j (N_j / M) * (1 / b_j) * sum_{i in sample_j} g_i.
```

Conditional on the virtual table and the stratum allocation, `g_hat` is an
unbiased estimator of `g_full`. This statement concerns the gradient before
Adam; Adam's nonlinear moment update is not itself an unbiased estimator of a
full-gradient Adam update.

Operationally:

1. forward all `M` latent samples without activations;
2. construct the global rank table;
3. select at most `B_back=128` rank indices with mandatory active-stratum
   coverage;
4. rerun only those latents with activations;
5. weight contributions by `N_j/(M b_j)` and take one Adam step.

The nominal generator-evaluation cost becomes `M+B_back`, rather than `2M`.
For `M=1024`, that is a 43.75% reduction in examples for a global update before
accounting for the controller's shorter global duration.

### 6.6 Final paper-only phase

The last 30% remains the exact paper Algorithm-2 update at the frozen local
bandwidth used by the present candidate. No occupancy pulse is allowed to
interrupt it in the first mechanism experiment.

This constraint has two purposes:

- it isolates whether the new controller and estimator preserve the known
  finishing behavior;
- it prevents an empirical support heuristic from being confused with a new
  population-identifiability theorem.

## 7. Why this follows from the formalization without overclaiming

The formal project established that identifiability depends on whether the
interaction field can distinguish admissible distribution directions and that
finite-basis stability depends on quantitative conditioning, not merely an
abstract zero-set statement.

OA-SQD converts those lessons into an empirical design rule:

- target atlas regions are observable finite cells;
- a cell with near-zero minibatch occupancy is poorly observed, irrespective
  of whether the population field is identifying;
- the controller spends computation until every certified target cell has
  usable generated support;
- positive stratum weights retain every direction rather than zeroing a rare
  cell;
- the paper-only finishing phase preserves the original local field once the
  empirical finite-cell system is no longer under-resolved.

What is **not** implied:

- finite atlas occupancy is not equality of probability measures;
- a switching policy is not automatically a gradient flow;
- conditional gradient unbiasedness does not prove neural-network convergence;
- an empirical partition is not the same object as the paper's analytic
  interaction basis;
- a one-dimensional rank result does not justify a high-dimensional claim.

## 8. Implementation order

### Stage O0 -- seal old evidence and create a new registry

Before code changes:

1. treat both `lbqcd_development_registry.json` and
   `lbqcd_confirmatory_registry.json` as permanently read-only;
2. create `oasqd_development_registry.json` with new target parameters and new
   random target-construction seeds;
3. record its SHA-256 in an OA-SQD development protocol;
4. prohibit outcome-based edits to old registries and old result artifacts.

Acceptance: a manifest test fails if an OA-SQD development target exactly
matches an old frozen target specification.

### Stage O1 -- atlas only

Implement and test the tail-aware atlas before training a generator.

Required controls:

- separated tail mixtures with masses `.05`, `.02`, `.01`, `.005`, `.0025`;
- separated interior mixtures;
- Student-t controls over several degrees of freedom;
- log-normal/skewed connected controls;
- overlapping mixtures with no stable gap;
- contamination targets where the remote mass is legitimate;
- split-half and bootstrap reproducibility.

Report boundary precision, mass error, false-positive rate, unresolved rate,
and atlas-size sensitivity. Do not select atlas thresholds by downstream ED2
at this stage.

### Stage O2 -- state-aware stop with existing full RSR

Hold `M=1024` and the full RSR backward pass fixed. Compare:

- QLD-v1;
- frozen fixed-duration LB-QCD;
- target-routed RSR with occupancy stopping;
- occupancy-triggered pulses with hysteresis.

This is the highest-value causal experiment. It tests the early-help/late-harm
hypothesis without confounding it with a new gradient estimator.

Promotion criterion: improve endpoint ED2 relative to frozen LB-QCD while
retaining or improving coverage time and using fewer global updates in both
initialization regimes.

### Stage O3 -- stratified backward pass

On fixed virtual tables:

1. compute the exact full-RSR gradient;
2. repeatedly sample the stratified estimator;
3. verify its empirical mean converges to the exact gradient;
4. compare variance against a uniform `B_back` subset;
5. verify every active stratum is sampled;
6. audit the exact `M+B_back` work count;
7. test tied outputs and strata with one rank.

Then compare full and stratified RSR under the same occupancy controller.

Promotion criterion: no meaningful ED2/coverage regression, a clearly lower
generator-evaluation count, and no gradient-bias signal at the resolution of
the Monte Carlo audit.

### Stage O4 -- adaptive `M` and stratified target table

Add the adaptive batch rule only after O2 and O3 succeed. Compare:

- iid versus randomized stratified target tables;
- fixed `M=1024` versus adaptive `M`;
- `k_target` values 4 and 8;
- atlas sizes 4,096, 8,192, and 16,384.

Report `resolution_cap_hit`, realized ranks per active region, target atlas
samples, global-forward examples, backward examples, and sort work separately.

### Stage O5 -- fresh development tournament

Use paired seeds and a small, predeclared candidate set. Baselines:

1. paper `tau=.5`;
2. the same per-cell paper bandwidth oracle as a diagnostic only;
3. QLD-v1;
4. frozen resolution-gated LB-QCD;
5. OA-SQD.

Suggested development gates:

| gate | proposed threshold |
|---|---:|
| OA-SQD / QLD-v1 ED2 | at most `.95` |
| OA-SQD / selected paper ED2 | at most `.78` |
| each primary initialization / QLD-v1 | at most `.98` |
| worst predefined family / QLD-v1 | at most `1.05` |
| coverage-event time / frozen LB-QCD | at most `1.00` |
| generator evaluations / paper | at most `3.0` |
| divergence | no worse than all principal baselines |

These are proposed development targets, not post hoc reinterpretations of the
old `.80` gate. Their purpose is to require enough margin to justify a new
confirmation.

### Stage O6 -- freeze a third, untouched confirmation

Only after one candidate is selected:

1. freeze code, hyperparameters, metrics, target registry, seeds, and gates;
2. use new target construction seeds and parameter combinations;
3. retain missing and concentrated starts;
4. add a small predeclared boundary-target group near the atlas resolution
   limit;
5. run the frozen candidate once;
6. report the conjunction gate exactly, even if one threshold narrowly fails.

Do not revisit the old confirmatory suite to tune OA-SQD.

### Stage O7 -- theory and Lean follow-up

Only after the mechanism survives development:

1. formalize conditional unbiasedness of the stratum-weighted stop-gradient
   estimator on a fixed finite virtual table;
2. prove that strictly positive quantile-stratum weights preserve the finite
   empirical zero set;
3. bound target-atlas approximation and occupancy-estimation error under an
   explicit finite partition;
4. keep controller convergence separate unless a genuine switched-dynamics
   argument is supplied.

No new axiom is needed for the finite estimator identity.

## 9. Fresh development registry design

The new registry should test mechanisms rather than recycle old coordinates.
Predeclare families such as:

- unequal tail mixtures with rare mass between `.003` and `.06`;
- unequal interior mixtures with the same mass range;
- equal mixtures with component counts not used previously;
- heteroscedastic mixtures with rare narrow and rare broad components;
- legitimate remote contamination;
- connected Gaussian, Student-t, skewed, and log-concave controls;
- overlapping mixtures where no quantile gap should be certified;
- two near-threshold cases where `resolution_cap_hit` is expected.

Use both missing-mode and concentrated starts. Keep far initialization as a
separate diagnostic until a mechanism is designed specifically for global
translation; do not dilute the main claim by silently changing that scope.

## 10. Evaluation and cost accounting

### Primary effectiveness

- target-balanced geometric mean of cell-median ED2 ratios;
- hierarchical target bootstrap with targets as the resampling unit;
- paired-seed win fraction;
- family and initialization ratios.

### Secondary effectiveness

- SW1;
- mass-L1;
- weighted-reach/coverage event time and censoring;
- endpoint quantile error by atlas region;
- fraction of target regions never occupied;
- divergence and extreme-output rate.

### Controller diagnostics

- number and duration of global pulses;
- state transitions and hysteresis reversals;
- false triggers on connected controls;
- deficits missed by the atlas;
- fraction of training in each state;
- stopping time relative to first correct occupancy.

### Complete computation ledger

- wall time;
- generator forward calls;
- generator example evaluations;
- unique latent draws;
- target atlas samples and per-update target samples;
- backward examples;
- sort work;
- paper kernel pairs;
- controller-probe evaluations;
- peak stored outputs and peak activation batch.

Report update-matched and generator-example-matched comparisons. The first
measures the quality ceiling; the second tests whether the algorithm is
actually more efficient.

## 11. Risk register and falsification tests

| risk | falsification test | response |
|---|---|---|
| Atlas labels heavy-tail extremes as a component | connected heavy-tail false-positive suite | tighten persistence/compactness; allow `unresolved` |
| Atlas misses overlapping rare mass | overlapping-mixture sensitivity | fall back to QLD; do not claim universal rare-mode detection |
| Occupancy is correct but local shape is poor | region-conditional W2/variance diagnostic | stop global pulses; rely on local/final phase |
| Controller chatters | state-transition count and ablation | two-check hysteresis and cooldown |
| Probe noise drives routing | repeat independent probes, confidence intervals | increase probe only when uncertainty demands it |
| Stratified gradient is biased | fixed-table Monte Carlo mean versus full gradient | fix weights before any training comparison |
| Mandatory rare ranks cause huge variance | per-stratum contribution variance | clipped Neyman-style allocation or control variate |
| Empirical atlas overfits target sample | atlas-size and resampling study | count atlas budget; widen intervals; new reservoir at confirmation |
| Adaptive `M` merely adds compute | example-matched comparison | require compute gate, not only endpoint improvement |
| Paper phase undoes occupancy | phase-boundary and endpoint occupancy | test a short quantile pulse before, never during, final paper block |
| Improvement is old-suite overfitting | third untouched registry | abandon the claim if it does not transfer |

## 12. Priority and usefulness ratings

Scores are informed judgments, not measured effects.

| proposal | expected usefulness | implementation effort | scientific risk | priority |
|---|---:|---:|---:|---:|
| State-aware occupancy stopping | `9.5/10` | medium | low | 1 |
| Rank-stratified backward pass | `9/10` | medium | medium | 2 |
| Tail-aware persistent atlas | `8.5/10` | medium | medium | 1, prerequisite |
| Adaptive virtual batch | `8/10` | low-medium | low | 3 |
| Randomized stratified target table | `8/10` | medium | medium | 3 |
| Per-stratum control variate | `7/10` | medium-high | medium | later |
| Generated-output memory bank | `6/10` | high | high staleness risk | later |
| Mixed-divergence drifting branch | `6.5/10` | medium-high | kernel/ratio risk | parallel, not successor core |
| More fixed large-batch training | `3/10` | low | high compute/endpoint risk | reject |
| Unbalanced mass deletion | `2.5/10` | medium | conflicts with goal | diagnostic only |

## 13. Decision rule after the next pass

Proceed to a fresh confirmation only if the causal sequence is visible:

1. atlas validation shows acceptable tail sensitivity without heavy-tail false
   positives;
2. occupancy stopping improves the fixed-duration endpoint while preserving
   early coverage;
3. stratified backward reproduces the full gradient in the fixed-table audit
   and materially reduces generator evaluations;
4. adaptive `M` preserves those gains across region masses;
5. one frozen candidate passes every new development gate.

If O2 fails, the core early-help/late-harm hypothesis is wrong and the project
should pivot to the mixed-divergence or optimizer-dynamics branch. If O2 works
but O3 fails, keep state-aware full RSR as the quality candidate and treat
compute reduction as a separate problem. If the atlas fails on connected
tails, retain state-aware control using a label-free quantile residual without
claiming explicit component discovery.

## Bottom line

The current result already demonstrates a broad low-dimensional improvement
over the paper baseline, but the extra LB-QCD router is not yet an efficient or
uniform improvement over QLD-v1. The data says the next gain is most likely to
come from using global transport **only when the current generator lacks
target support**, then stopping it as soon as that support is restored.

OA-SQD is the most direct, falsifiable implementation of that lesson. It does
not ask a larger batch to solve every part of learning. It uses a large forward
table to see rare mass, a small stratified backward set to act on it, and the
paper field to finish.
