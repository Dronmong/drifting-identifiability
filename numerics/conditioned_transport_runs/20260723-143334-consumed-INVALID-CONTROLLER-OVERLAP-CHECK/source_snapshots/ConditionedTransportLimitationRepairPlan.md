# Conditioned transport limitation audit and repair plan

**Status:** post-confirmation diagnosis and proposed development plan  
**Date:** 2026-07-23  
**Primary confirmed result:** `ProjectionKernelOptimizationConfirmationResults.md`  
**Broader confirmation:** `NeuralConditionedTransportConfirmatoryResults.md`

## 1. Purpose and evidence boundary

This document records the limitations currently visible in the repository's
conditioned transport-then-amortize model and gives a concrete plan for
repairing them.

The present primary is:

- an exact persistent target quantile atlas;
- a conditioned direction bank;
- 32 active registered directions per transport macro-step;
- a global 512-particle projected-quantile correction;
- a normalized local paper-field correction with fixed weight `0.25`;
- 128 weighted positive and negative representatives for the local field; and
- neural amortization through eight 64-example student updates per
  512-particle teacher refresh.

The primary already passed a frozen 16-target synthetic confirmation against
the repository's optimized paper-model port:

| Metric | Candidate/paper geometric ratio | 95% interval | Wins |
|---|---:|---:|---:|
| ED2 | `0.3305` | `[0.2520, 0.4313]` | `15/16` |
| held-out SW1 | `0.5711` | `[0.4969, 0.6545]` | `16/16` |

It also matched the paper port's generator-example count and exact
kernel-pair count. Its measured CPU online-wall ratio was `0.6150`, and its
setup-plus-training ratio was `0.7636`.

Those results are not reopened by this audit. The artifacts are consumed.
Every diagnostic below is either:

1. already reported in a frozen result document; or
2. an explicitly post-hoc analysis of saved arrays/rows used only to design a
   new experiment.

No new performance claim may be made from the consumed registries. Any revised
algorithm must be selected on development data and evaluated on a new frozen
registry.

## 2. Executive conclusion

The central mechanism appears healthy. The persistent global projected-
quantile transport explains most of the quality gain and remains the most
reliable component.

The clearest limitations are in two auxiliary mechanisms:

1. the local paper-field weight is fixed even though the local correction is
   helpful only in some regimes; and
2. the 128-representative projection tree uses only seven fixed directions,
   even when the ambient dimension and registered direction bank are much
   larger.

These are favorable failure modes: they can be repaired without discarding
the confirmed architecture.

The recommended order is:

1. replace the representative tree with a geometry-aware construction;
2. make representative capacity conditional on measured approximation
   difficulty;
3. replace the fixed local weight with a cross-fitted validation controller;
4. add support-sensitive and scale-sensitive diagnostics;
5. combine the repairs only after isolated ablations succeed; and
6. run one new preregistered confirmation.

## 3. Current strengths that must be preserved

### 3.1 Global occupancy correction

The global 512-particle projected-quantile step is the main improvement over
the paper port. It sees a substantially larger coherent particle population
than a 64-example paper update and directly corrects projected occupancy.

The broader 64-target confirmation found exact-primary/paper geometric ratios
of:

- `0.3189 [0.2916, 0.3492]` for ED2; and
- `0.5517 [0.5277, 0.5772]` for held-out SW1.

The exact primary beat paper on both endpoints in `62/64` targets. Rare-mode
median coverage was `1.00`, versus `0.50` for paper, and rare-mass error was
`0.01002`, versus `0.02519`.

### 3.2 Direction-bank conditioning

The direction bank is not an arbitrary random collection. It is expanded
until the low-dimensional quadratic sensing matrix has full rank and bounded
condition number. This repaired the earlier 16-dimensional reversal.

Active-32 sharding then reduced repeated projection work while retaining the
registered full atlas and balanced exposure.

### 3.3 Coherent transport and amortization

The first student gradient after a teacher refresh is checked against the
corresponding rank-loss gradient. The later microsteps are honestly treated as
frozen-teacher distillation, not as exact reranking.

### 3.4 Reproducible evaluation

The current runner records source snapshots, registries, outputs, references,
hashes, ledgers, and deep endpoint recomputation. The next experiment must
retain this standard.

## 4. Limitation A: the fixed local correction is not universally useful

### 4.1 Observed behavior

A post-hoc comparison of the canonical dense-support exact global and exact
hybrid rows shows that the fixed local correction has strongly
regime-dependent value.

It helps:

- every tested 8D target;
- every tested 16D target; and
- all four rare-mixture dimensions.

It hurts several easier/common cases:

- 2D balanced GMM;
- 2D correlated heavy tail;
- 2D nonlinear target;
- 4D correlated heavy tail; and
- especially the 4D nonlinear target.

Grouped geometric hybrid/global ratios from the consumed canonical rows were:

| Dimension | ED2 hybrid/global | Hybrid wins |
|---:|---:|---:|
| 2 | `0.906` | `1/4` |
| 4 | `0.925` | `2/4` |
| 8 | `0.622` | `4/4` |
| 16 | `0.694` | `4/4` |

The aggregate ratios remain below one because the large improvements outweigh
the failures. This does not make a fixed local coefficient optimal.

### 4.2 Likely mechanism

The global term directly attacks projected distributional mismatch. The local
paper field supplies geometric information missed by a finite projection
bank, but it is not aligned with the global correction on every target and at
every stage of training.

RMS normalization controls magnitude, not direction. A normalized local
correction can still:

- move particles away from globally correct ranks;
- increase off-support bridge mass;
- over-correct an already easy low-dimensional target; or
- trade one tail/mode improvement for a worse common-region fit.

The existing one-step guard is not enough. Its same-atlas criterion does not
reliably predict the final held-out neural endpoint.

### 4.3 Repair: cross-fitted local-weight controller

Split projected information into three immutable groups:

1. **transport directions:** used by the global teacher;
2. **controller directions:** used only to choose a local weight; and
3. **evaluation directions:** never accessed until final evaluation.

At every teacher refresh:

1. compute the global correction `c`;
2. compute the normalized local correction `v` once;
3. form candidate teacher particles with local weights
   `{0, 0.05, 0.10, 0.25}`;
4. score all candidates on the controller directions;
5. reject candidates that worsen protected tail/support diagnostics beyond a
   frozen tolerance;
6. choose the smallest weight whose controller score is within a frozen
   tolerance of the best admissible score; and
7. amortize only the selected teacher.

Choosing the smallest near-optimal weight is a trust-region rule: it limits
unnecessary local perturbation without forbidding the strong high-dimensional
and rare-mode corrections observed in the current model.

The controller score should begin with a simple predeclared quantity:

\[
J_{\mathrm{ctrl}}
  = \operatorname{SW1}_{\mathrm{controller}}
    + \alpha_\mu E_\mu
    + \alpha_\Sigma E_\Sigma,
\]

where the moment terms are normalized target-relative errors. Tail and
support conditions should initially be rejection constraints rather than
additional freely tuned weights.

The evaluation direction bank must never influence the selected coefficient.

## 5. Limitation B: representative compression ignores most high-dimensional geometry

### 5.1 Concrete implementation issue

The current `projection_tree_representatives` implementation sets:

```python
levels = count.bit_length() - 1
selected_directions = unit_directions[
    torch.arange(levels) % len(unit_directions)
]
```

For `M = 128`, the tree has seven binary levels. Consequently the entire
512-to-128 compression uses only the first seven registered directions,
including in 16 dimensions with a 176- or 192-direction conditioned bank.

The tree guarantees equal occupancy—four observations per leaf—but does not
minimize cluster radius, field error, or kernel error.

### 5.2 Observed consequences

The frozen confirmation reports:

- median local-field relative-L2 error `0.0804`;
- minimum field cosine `0.8915`;
- median row/column mass relative-L2 errors `0.0389 / 0.1087`; and
- largest target-level mean mass errors `0.3295 / 0.3116`.

Post-hoc dimension grouping shows the qualitative pattern:

- `M=128` improves or regularizes the endpoint in 2D and 4D;
- it becomes approximately neutral around 8D; and
- it slightly worsens the dense local-field endpoint in 16D.

Representative radii and field error grow with dimension. This is consistent
with a tree that uses seven predetermined axes instead of the directions that
best resolve each leaf.

### 5.3 Repair B1: per-node variance-aware registered-direction tree

Precompute all registered projections once:

```text
P[i,l] = <point_i, direction_l>.
```

For each current leaf:

1. compute the projected variance of the leaf along every admissible
   registered direction;
2. choose the direction with maximum variance;
3. use deterministic index order to break exact ties;
4. stably split the leaf at its median projection;
5. continue until `M` nonempty leaves exist;
6. use each leaf mean as its representative; and
7. retain the exact integer multiplicity.

This preserves:

- deterministic behavior;
- equal occupancy when `N/M` is integral;
- exact support mass;
- current weighted-field semantics;
- generated self-mask deletion through the assignment map; and
- compatibility with the registered direction bank.

It changes only which registered direction splits each leaf.

### 5.4 Repair B2: radius-priority refinement

A second variant should choose which leaf to refine according to its current
RMS or maximum Euclidean radius. The split direction is still the
maximum-variance registered direction within that leaf.

This permits nonuniform leaf capacity. Large or geometrically complex regions
receive more representatives than compact regions.

Multiplicity remains exact, but leaf populations need no longer be equal.
All code assuming a constant multiplicity must therefore be audited.

The first comparison should keep the equal-occupancy version as the safer
primary and treat radius-priority refinement as a development ablation.

### 5.5 Repair B3: unsupervised tail protection

The algorithm must not use known synthetic component labels. A deployable
tail rule can instead use projected extremeness.

For every support point, compute a robust score such as:

\[
s_i=\max_\ell
  \frac{\left|P_{i\ell}-\operatorname{median}(P_{\cdot\ell})\right|}
       {\operatorname{IQR}(P_{\cdot\ell})+\epsilon}.
\]

Reserve a small frozen fraction of representatives for the most extreme
points, deduplicate selections, and construct the variance tree on the
remaining mass.

This is intended to prevent small tail or rare groups from being averaged
into common leaves. It is not a guarantee that every semantic mode receives a
representative, and must not be described that way.

### 5.6 Repair B4: dimension/error-adaptive capacity

Do not force `M=128` in every dimension.

The development comparison should test:

- `M=128` in all dimensions;
- `M=256` in 8D and 16D; and
- dense `M=512` as the reference.

A later deployed rule may choose the smallest capacity satisfying frozen
radius/mass proxy thresholds. Direct dense-field error is appropriate for
development audits but should not be required at every deployed step, because
computing it would erase the intended kernel saving.

## 6. Limitation C: equal occupancy does not guarantee rare-mode retention

In the cost-repaired confirmation, the candidate's median rare-component mass
error was `0.0240`, versus `0.0500` for paper. Nevertheless, both algorithms
missed the rare component in the 16D rare-GMM cell.

The current tree cannot distinguish:

- a small genuine component;
- an elongated tail;
- a few outliers; and
- a sparse bridge.

This should be addressed with:

1. tail-protected representatives;
2. rare-weight stress tests at `0.5%`, `1%`, `2%`, `5%`, and `10%`;
3. several target realizations at each weight; and
4. reporting both coverage and mass calibration.

Increasing the global particle population is deliberately not the first
repair. The 512-particle global teacher is one of the successful mechanisms
and should remain fixed until the representative/local-controller changes are
understood.

## 7. Limitation D: current endpoints can miss off-support bridge mass

Visual inspection of some balanced-mixture outputs shows candidate particles
between clusters even when ED2 and held-out SW1 improve.

This is plausible because:

- a finite collection of projected marginals does not fully constrain joint
  support geometry;
- ED2 and averaged SW1 can reward broad global corrections while tolerating a
  smaller local precision defect; and
- a continuous MLP generator can interpolate between separated modes.

Post-hoc nearest-neighbor diagnostics on frozen outputs suggest that the
candidate often gains recall at a small precision cost on rare, nonlinear,
and correlated targets. The balanced-mixture case is less favorable. These
diagnostics are exploratory and were not registered endpoints.

### 7.1 Required metric panel

The next protocol should include:

- ED2;
- held-out SW1;
- target-calibrated support precision;
- target-calibrated support recall;
- off-support occupancy;
- mean error;
- covariance Frobenius error;
- mixture-mode coverage where labels exist; and
- rare-component mass error where labels exist.

For support precision/recall, determine the neighborhood threshold using an
independent target-versus-target sample split. Do not tune the radius on
candidate outputs.

Known component labels may be used for evaluation on synthetic mixtures, but
not by the training algorithm or representative builder.

### 7.2 Conditional architectural escalation

Do not immediately replace the generator.

If the geometry-aware tree and local controller preserve ED2/SW1 gains but do
not reduce bridge occupancy, then test:

1. a small residual direction supplement chosen on training residuals and
   frozen before evaluation;
2. structured nonlinear features used only by the controller; and finally
3. a mixture/discrete-latent or mixture-of-experts generator.

The mixture-generator experiment is a later architecture study because it
changes capacity and may itself explain improvements.

## 8. Limitation E: scale and bandwidth sensitivity are not established

The main experiments normalize targets and use a fixed local bandwidth
`tau = 1`. This can hide sensitivity to scale and local geometry.

The robustness matrix should include target scales:

```text
0.5x, 1x, 2x, 4x
```

Compare:

1. fixed `tau = 1`;
2. a target-derived bandwidth frozen from an independent target sample, such
   as a quantile of pairwise distances; and
3. a small multiscale field with predeclared bandwidths and normalized
   combination weights.

The bandwidth statistic must be selected before model outcomes are observed.
This experiment should not be combined with representative or controller
selection until the isolated repairs are settled.

## 9. Limitation F: computational and statistical scaling remain incomplete

The current result does not establish:

- dimensions above 16;
- image or learned-encoder features;
- GPU throughput;
- peak device memory;
- amortization of one target atlas over several trained generators;
- behavior under a smaller target pool; or
- robustness over multiple optimized target realizations.

The exact atlas reads 20,480 target points before training and the local field
uses another 10,240 target examples. The resulting target-access ratio is
`1.5` relative to the paper port.

The low-dimensional quadratic direction certificate also scales as
`d(d+1)/2` and cannot be used literally at image-feature dimensions.

After the low-dimensional repair confirmation, the scalable study should
measure:

- atlas sizes `2k`, `5k`, `10k`, and `20k`;
- amortization over `1`, `2`, `4`, and `8` generators sharing one atlas;
- dimensions `32`, `64`, and `128`;
- exact CPU and GPU wall time;
- peak memory;
- direction projections and sorting work; and
- serialized atlas size.

At higher dimensions, use a first-order tight frame plus held-out empirical
diagnostics, not the impossible ambient full-covariance sensing matrix.

## 10. Implementation stages

### Stage 0: freeze and reproduce the diagnosis

Add a read-only analysis script that consumes the existing canonical
artifacts and emits:

- hybrid/global ratios by dimension and family;
- compressed/dense ratios by dimension and family;
- representative radii by dimension;
- field/mass approximation errors by dimension;
- rare-mode outcomes; and
- exploratory support precision/recall.

The script must record all input artifact paths and hash its output.

No revised model should be run until the diagnostic report reproduces.

### Stage 1: representative construction

Implement:

```text
projection_tree_strategy =
    fixed-level
  | variance-per-node
  | radius-priority
  | variance-with-tail-reserve
```

Required regression tests:

1. every original point has exactly one assignment;
2. multiplicities sum to support size;
3. multiplicities equal assignment counts;
4. no leaf is empty;
5. centers are leaf means;
6. deterministic replay is bitwise stable;
7. stable behavior under input permutations away from exact ties;
8. `M=N` reproduces the exact local field and one-step parameters;
9. self-mask deletion still removes the correct conceptual negative entry;
10. weighted mass identities remain valid; and
11. all outputs remain finite under extreme scale.

The development run changes only the tree strategy and `M`.

### Stage 2: isolated representative experiment

Use the consumed development registry, never the frozen confirmation registry
as a selection set.

Arms:

```text
current fixed-level M128
variance-per-node M128
radius-priority M128
tail-protected variance M128
variance-per-node M256 (8D/16D)
dense M512 reference
```

Primary mechanism outcomes:

- field relative-L2 error;
- field cosine;
- row/column mass error;
- RMS/max radius; and
- rare-mode retention.

Secondary endpoints:

- ED2;
- held-out SW1;
- support precision/recall;
- wall time; and
- exact cost ledgers.

Select a representative method only if it improves the approximation
mechanism and does not merely exploit endpoint noise.

### Stage 3: cross-fitted local controller

Hold the selected representative builder fixed.

Arms:

```text
global only
fixed local 0.25
existing tail guard
cross-fitted discrete local controller
```

Required tests:

1. evaluation directions are never accessed during training;
2. weight `0` is exactly global-only;
3. adversarial local fields are rejected;
4. an aligned synthetic local field selects a positive weight;
5. controller decisions replay deterministically;
6. controller projection cost is recorded; and
7. protected tail constraints fail closed.

Report the selected-weight distribution by macro-step, family, and dimension.
The controller is successful only if it retains the high-dimensional/rare
benefit and removes most low-dimensional regressions.

### Stage 4: combine only verified repairs

Combine the representative and local-controller revisions only if both
succeed in isolation.

Run a factorial check:

| Representatives | Local policy |
|---|---|
| current | fixed |
| repaired | fixed |
| current | cross-fitted |
| repaired | cross-fitted |

This prevents crediting one repair for the other and detects an adverse
interaction.

### Stage 5: freeze a new confirmation protocol

Before generating new targets, freeze:

- exact source hash;
- model and optimizer settings;
- representative strategy and capacity rule;
- controller directions and candidate weights;
- support metric definitions;
- primary and secondary outcomes;
- bootstrap procedure;
- success/non-inferiority thresholds;
- target families and dimensions;
- rare weights and scales; and
- all random seeds except the unrevealed target-registry master seed.

Use at least two new target realizations per dimension/family cell and both
concentrated and broad initializations.

Required comparators:

1. optimized paper port;
2. current confirmed active-32 `M=128` model;
3. global-only conditioned transport;
4. dense fixed-local conditioned transport; and
5. the repaired candidate.

### Stage 6: high-dimensional stress test

Only after the fresh low-dimensional repair confirmation:

- test dimensions `32`, `64`, and `128`;
- separate atlas setup from online cost;
- report CPU/GPU wall and peak memory;
- include an atlas-size curve;
- retain a disjoint evaluation bank; and
- avoid claiming image-scale performance.

## 11. Proposed decision gates

Exact numerical thresholds must be frozen before a new run. The following are
recommended starting gates.

### 11.1 Representative mechanism gate

Relative to the current `M=128` tree:

- lower median field relative-L2 error in both 8D and 16D;
- higher minimum or lower-tail field cosine;
- no material increase in row/column mass error;
- lower RMS and maximum representative radius;
- no lost rare-mode coverage; and
- no more than a frozen small endpoint degradation in any dimension.

If geometry improves but neural endpoints do not, retain the simpler current
tree and conclude that representative error is not the active endpoint
bottleneck.

### 11.2 Local-controller gate

Relative to fixed weight `0.25`:

- retain aggregate ED2 and SW1 quality;
- improve the low-dimensional cells where fixed local currently hurts;
- retain high-dimensional and rare-mode gains;
- do not worsen support precision or rare-mass calibration; and
- record all extra projection/controller work.

If the controller selects zero almost everywhere, simplify the model to global
transport rather than preserving an inactive mechanism.

### 11.3 Confirmation gate

The repaired model must:

- retain an ED2 and held-out-SW1 confidence-interval upper bound below one
  against paper;
- improve the predeclared primary aggregate against the current confirmed
  model;
- be non-inferior on the other primary endpoint;
- retain rare-mode coverage;
- reduce representative field error or fixed-local failures as intended; and
- report, rather than hide, any support-precision tradeoff.

The comparison against the current confirmed model is essential. Beating
paper again is not enough to establish that these revisions improved the
model.

## 12. Failure interpretation

The experiment should produce useful conclusions even if a repair fails.

| Outcome | Interpretation | Next action |
|---|---|---|
| Better tree field error and better endpoints | Compression was a real bottleneck | Promote repaired tree |
| Better tree field error, unchanged endpoints | Dense/local accuracy is not the current quality bottleneck | Retain simpler tree |
| Controller helps low dimensions and retains rare/high-D gains | Fixed local weight was the main avoidable defect | Promote controller |
| Controller usually selects zero | Global transport is sufficient in this scope | Remove local term |
| ED2/SW1 improve but bridges remain | Generator/support geometry is the next bottleneck | Test residual directions, then mixture generator |
| Rare failure remains after tail protection | Global sample occupancy or generator capacity is limiting | Test larger particle population or stratified latent architecture |
| Scale stress fails | Fixed bandwidth/normalization is over-specialized | Promote frozen adaptive or multiscale bandwidth study |
| High-D cost dominates | Low-D direction certificate does not scale | Move to structured fast transforms/intrinsic subspaces |

## 13. Guardrails

1. Do not alter or relabel consumed confirmation artifacts.
2. Do not use evaluation directions for controller decisions.
3. Do not use mixture labels in the training algorithm.
4. Do not describe equal-occupancy leaves as semantic mode protection.
5. Do not describe representative fields with `M<N` as exact.
6. Do not choose `M`, local weight, bandwidth, or support radius from final
   test outcomes.
7. Do not equate matched kernel pairs with matched FLOPs.
8. Keep target access, projections, sorting, storage, wall time, and memory
   visible.
9. Keep exact-atlas and randomized KLL conclusions separate.
10. Do not generalize synthetic dimensions 2--16 to images or arbitrary
    high-dimensional features.

## 14. Recommended immediate move

The next implementation should be **Stage 0 plus Stage 1**, followed by the
isolated representative experiment.

This is the highest-confidence improvement because:

- the implementation defect is explicit;
- its error grows in the same regime where compression stops helping;
- the repair preserves the successful global architecture;
- it can be audited directly against the dense field; and
- it does not require additional model capacity or target labels.

After that result is understood, implement the cross-fitted local controller.
Do not combine both revisions in the first experiment.
