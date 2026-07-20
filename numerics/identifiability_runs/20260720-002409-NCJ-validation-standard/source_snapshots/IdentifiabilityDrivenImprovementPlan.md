# Identifiability-driven algorithm improvement plan

Status: proposed next performance program, 2026-07-19.

This is the single implementation plan for the next empirical leg of the
project. It does not revise or overwrite the completed low-dimensional studies.
Those remain the audit trail that motivated this proposal.

The objective is deliberately narrower than reproducing the paper's ImageNet
experiments:

> Construct a modification of the original Algorithm 2 estimator, motivated
> directly by the repository's formal results, and establish a general
> improvement over a strong exact-paper implementation on fresh, held-out
> low-dimensional distribution families under matched architecture and
> compute.

"General" here means a target-balanced improvement across multiple target
families, dimensions, and initialization regimes. It does not mean a theorem
that one discrete optimizer dominates another for every probability measure,
and it does not mean superiority over the paper's published image benchmarks.

## 1. Why a new algorithmic candidate is needed

The fresh attribution study in `LowDimAttributionResults.md` rejected the
current global policy:

* geometry-matched bandwidth did not beat the tuned fixed bandwidth;
* scaled step size did not beat the tuned fixed step;
* disabling the eye mask helped ring/circle/moon targets substantially;
* the cluster-count mask trigger misfired on an overclustered Gaussian target;
* the pre-registered aggregate gate failed.

Therefore the cluster-count mask policy is not the candidate to transfer into
a learned generator. The next candidate must address the three distinct
failure mechanisms already diagnosed by the formalization and numerics:

| Failure mechanism | Certified or measured diagnosis | Required repair |
|---|---|---|
| Off-support freezing | The raw Algorithm 2 field is an affinity-mass product times a well-directed centroid signal; the mass product can become exponentially small | Remove or floor the identifiability-inert gain |
| Finite-batch self-mask distortion | The diagonal mask changes the finite estimator and can be harmful at low effective occupancy | Use an independent model reference batch rather than a heuristic diagonal mask |
| Homogeneous-swarm transport | Gain replacement moves a collapsed swarm but does not make it divide among target modes | Inject per-particle diversity without changing the identifiable population target |

The proposed method addresses these failures separately so that each effect can
be ablated.

## 2. Candidate: normalized, cross-fitted, jittered drifting

Working name: **NCJ drifting** (normalized, cross-fitted, jittered drifting).
The name is provisional and should not be used as a novelty claim until the
literature and experiments support one.

### 2.1 Exact Algorithm 2 factorization

For query/anchor `i`, let

```text
P_i      = total positive affinity mass
Q_i      = total negative affinity mass
Cpos_i   = positive self-normalized affinity centroid
Cneg_i   = negative self-normalized affinity centroid
Delta_i  = Cpos_i - Cneg_i
```

The audited identity in `Algorithm2Estimator.lean` is

```text
Vpaper_i = (P_i * Q_i) * Delta_i.
```

`Delta_i` is the distribution-matching signal. `P_i * Q_i` is a positive
per-query gain. It affects conditioning and speed, but not the pointwise zero
set when the masses are positive.

### 2.2 Normalized gain

The first new field is

```text
Vnorm_i = g_i * Delta_i,
```

with a strictly positive bounded gain

```text
0 < g_min <= g_i <= g_max.
```

Version 1 must use a constant gain `g_i = 1`. Its global scale is controlled by
the optimizer learning rate or a global norm clip. Do not begin with the
previous certificate-based gain: S7 found that the simple constant gain was
more reliable on the toy problem.

Permitted later gains must remain positive. A safe clipped form is

```text
g_i = clip(g_global * reliability_i, g_min, g_max),
```

where `reliability_i` may depend on effective sample size (ESS) or a certified
centroid-error proxy. This is an ablation after the constant-gain result, not a
part of the initial candidate.

Implementation rule: compute `Delta_i` directly from the two centroids. Do not
numerically divide the raw drift by a potentially tiny `P_i * Q_i`.

### 2.3 Cross-fitted model reference

The query batch and the negative/model-reference batch must be generated from
independent latent draws:

```text
x_query = G(z_query)
y_neg   = G(z_reference),       z_query independent of z_reference.
```

The target batch `y_pos` is independently sampled from the data law. Because a
query is not reused as its own negative key, there is no distinguished diagonal
entry and no eye mask is applied.

This intervention has three purposes:

1. eliminate the small-occupancy diagonal-mask artifact;
2. match the fixed-anchor/no-mask statistical object already analyzed in the
   repository;
3. make the realized negative centroid a genuine independent empirical
   estimate of the current model law.

Cross-fitting may require an additional generator forward pass. Every study
must therefore report both kernel-pair cost and total wall time. A
compute-matched paper baseline must receive the same forward-pass budget (for
example through a larger batch or additional update, fixed before testing).

### 2.4 Symmetric Gaussian jitter

Before computing affinities, independently jitter both empirical laws:

```text
y_pos_tilde = y_pos + sigma * eps_pos
y_neg_tilde = y_neg + sigma * eps_neg
x_tilde     = x_query + sigma * eps_query

eps_pos, eps_neg, eps_query iid standard Gaussian.
```

The update target is computed at `x_tilde`. Since `x_tilde` is an additive
perturbation of `x_query`, its derivative with respect to the generated query
is the identity when used in a differentiable implementation.

At population level this compares the smoothed laws

```text
p_sigma = p convolved with N(0, sigma^2 I)
q_sigma = q convolved with N(0, sigma^2 I).
```

The rationale is not merely heuristic:

1. the Laplace converse identifies `p_sigma` and `q_sigma` from zero drift;
2. Gaussian convolution is injective, so `p_sigma = q_sigma` implies `p = q`;
3. independent jitter gives coincident generated particles different local
   probes and therefore activates splitting directions that a homogeneous
   deterministic swarm cannot select.

Candidate validation values are

```text
sigma / tau in {0, 0.10, 0.25, 0.50}.
```

One value must be selected on validation targets and frozen. An annealed noise
schedule is deferred until a fixed-ratio candidate has been evaluated. A
schedule adds a second causal question and must not be silently introduced.

### 2.5 Version-1 algorithm

For each update:

1. Draw independent `z_query`, `z_reference`, and target samples.
2. Compute `x_query`, `y_neg`, and `y_pos`.
3. Add independent symmetric Gaussian jitter to all three collections.
4. Build the exact paper row/column affinity on the jittered collections,
   without an eye mask.
5. Extract `Cpos_i`, `Cneg_i`, and `Delta_i`.
6. Use the constant-gain target `V_i = Delta_i`.
7. Apply a global learning rate and a pre-declared global vector-norm clip.
8. Log masses, ESS, gain, drift norm, jitter, and all distributional metrics.

The bandwidth is initially the tuned fixed-paper baseline bandwidth. Adaptive
bandwidth is not part of version 1.

## 3. Formal obligations

No new axiom is authorized by this plan. Every new Lean declaration must be
axiom-free and pass the existing trust audit.

### T1. Positive-gain zero-set preservation

Package the existing factorization and positive probe-scaling machinery into a
named theorem for Algorithm 2:

```text
(forall i, 0 < g i) ->
ZeroDrift normalizedCentroidField p q <->
ZeroDrift (positiveGainField g) p q.
```

For a finite frame statement, reuse
`interactionFrameBound_of_positiveGain`. State all finite-index and positive
minimum hypotheses explicitly.

Acceptance:

* no conditional Gaussian/RKHS axiom;
* `#print axioms` shows only kernel-level logical primitives;
* the theorem does not assume injectivity or the desired conclusion.

### T2. Jittered identifiability composition

Prove a clean Euclidean corollary of the form

```text
ZeroDrift (meanShiftDrift (laplaceKernel tau))
  (p convolve gaussian_sigma) (q convolve gaussian_sigma)
-> p = q.
```

The proof should compose:

1. `laplaceZeroDrift_identifies_euclidean` for the smoothed laws;
2. the certified Gaussian-convolution injectivity theorem.

State probability, Borel, bandwidth, dimension, and positive-variance
hypotheses explicitly. If the local convolution API uses a different order or
normalization, prove the bridge rather than changing the mathematical claim.

### T3. Cross-fitted estimator consistency

Instantiate the existing no-mask SNIS results with independent query and model
reference batches. The theorem should identify the population target of the
two centroids and provide a mean-square or deviation bound with explicit batch
size, weight floor/ESS surrogate, and radius assumptions.

Do not claim that independence makes a self-normalized ratio exactly unbiased.
It removes the diagonal self-pair and makes the existing fixed-weight analysis
applicable; ratio bias remains a separate finite-sample question.

### T4. Quantitative no-freeze statement

Formalize the conditioning distinction exposed by S7. A suitable theorem
schema is:

```text
paper:      norm Vpaper <= 2 * P * Q * R
normalized: norm Vnorm = g * norm Delta
```

Under `g >= g_min > 0` and `norm Delta >= c > 0`, normalized speed has a
positive floor `g_min * c`. Combine this with an explicit off-support upper
bound on `P * Q` for the Laplace kernel to state the exponential attenuation
of the paper gain.

This does not by itself prove global convergence. Its purpose is to certify
that the proposed preconditioner removes one measured exponential stall
mechanism.

### T5. Jitter and fission mechanism (follow-up)

After the numerical ablation confirms that jitter improves splitting, connect
it to the existing fission-instability results. The intended statement is
local and finite-particle:

* a homogeneous configuration receives identical deterministic directions;
* nondegenerate centered jitter produces distinct probes almost surely;
* on the certified fission-unstable neighborhood, the expected separation
  after one small step is positive.

Do not block the initial empirical experiment on T5. T1--T3 are the trust
contract required before promoting NCJ as an identifiability-preserving
algorithm; T4--T5 are the explanatory quantitative layer.

## 4. Software architecture

Use new files and do not modify historical run artifacts:

```text
numerics/identifiability_drift.py
    reusable paper and NCJ field implementations

numerics/run_identifiability_improvement.py
    validation, frozen test, and artifact generation

numerics/IdentifiabilityImprovementProtocol.md
    generated/frozen protocol containing exact target registries and hashes

numerics/IdentifiabilityImprovementResults.md
    results written only after the test run completes

numerics/identifiability_runs/<run-id>/
    manifests, source snapshots, rows, trajectories, final particles
```

`identifiability_drift.py` should reuse the audited affinity implementation in
`lowdim_drift.py` rather than reimplementing the paper estimator from scratch.
If refactoring is needed, first add agreement tests showing bitwise or
tolerance-level equivalence with the frozen historical implementation.

### Required field API

The library should expose conceptually:

```python
compute_paper_field(
    queries, positives, negatives, tau, mask, ...
)

compute_centroid_field(
    queries, positives, negatives, tau, gain="constant", ...
)

compute_ncj_field(
    queries, positives, negatives, tau, jitter_sigma,
    gain="constant", cross_fitted=True, ...
)
```

Every call should optionally return diagnostics:

```text
P, Q, P*Q
Cpos, Cneg, Delta
ESSpos, ESSneg
self-affinity leverage when applicable
gain and unclipped/clipped field norms
```

### Unit and invariant tests

Before any benchmark run, test:

1. `gain="paper"` reproduces the exact existing paper field.
2. `gain="power", gamma=1` reproduces paper gain.
3. Constant gain returns exactly `Cpos - Cneg`.
4. If positive and negative batches are identical and unmasked, every gain
   mode returns zero to numerical tolerance.
5. Cross-fitted mode contains no index-dependent diagonal operation.
6. `sigma=0` is bitwise identical to the corresponding no-jitter arm.
7. Jitter streams are independent but reproducible from recorded seeds.
8. Every trajectory row records setup-inclusive kernel pairs and wall time.
9. No NaN/Inf is silently converted into a finite metric.

## 5. Experimental program

### E0. Freeze the exact-paper baseline

Use the already tuned exact Algorithm 2 baseline:

```text
tau = 0.35
eta = 0.0525
paper eye mask = on
```

Reconfirm it on a small smoke registry without retuning. Preserve the current
normalization and feature scale. The baseline must be the exact paper-style
estimator, not the simpler SNIS field.

In addition to the exact baseline, define a compute-matched baseline that
receives the same generator-forward budget as cross-fitting. Report both; the
exact baseline is primary and the compute-matched baseline guards against an
extra-compute explanation.

### E1. Mechanism-isolating factorial

Run these six primary arms with paired seeds:

| Arm | Gain | Negative reference | Jitter |
|---|---|---|---|
| Paper | `P*Q` | reused/masked | none |
| Normalized only | constant | reused/masked | none |
| Cross-fit only | `P*Q` | independent/unmasked | none |
| Jitter only | `P*Q` | reused/masked | selected `sigma` |
| Normalized + cross-fit | constant | independent/unmasked | none |
| NCJ combined | constant | independent/unmasked | selected `sigma` |

If applying jitter to a reused/masked batch makes the diagonal semantics
ambiguous, retain the original index mask for the `jitter only` arm and record
that it masks the paired pre-jitter identity. The combined arm has no mask.

The validation factorial may cross all four `sigma/tau` values. After
selection, the test phase contains only the frozen `sigma`.

### E2. Fresh target registry

No configuration used in historical D3 or the fresh mask-attribution study may
appear verbatim. Create validation and test registries before running either.
They must use disjoint target parameters and master seeds.

Include at least these families:

* unimodal Gaussian and Student-t controls;
* equal- and unequal-weight Gaussian mixtures;
* separated mixtures with missing-mode initialization;
* rings or annuli;
* two-moon/circle-type curved supports;
* banana, spiral, and sine/ribbon connected supports;
* one heavy-tailed or contaminated target;
* dimensions `d in {1, 2, 5}`.

Each family must appear under multiple initialization regimes:

```text
covered      broad initialization covering the target
missing      one or more target components absent initially
far          cloud translated several target scales away
concentrated low-spread or nearly collapsed cloud
```

Synthetic truth may be used to evaluate coverage and mass error, but no true
component count, target label, or target geometry may enter the NCJ decision
rule.

Recommended standard profile:

```text
validation: >= 12 target configurations, 8 paired seeds per cell
test:       >= 16 new target configurations, 20 paired seeds per cell
particles:  N in {32, 64}; include a smaller-N stress subset
target batch B in {32, 64}; include one independent B sensitivity subset
steps:      enough to resolve finite-horizon and censored convergence
```

Use a smaller smoke profile only for code validation. Smoke results never
support a scientific claim.

### E3. Metrics

Primary distributional metric:

* energy distance squared (ED2).

Required secondary metrics:

* sliced Wasserstein-1;
* mode or support coverage;
* component-mass error where ground truth components exist;
* on-support drift residual;
* time to a pre-declared error threshold, with censoring;
* divergence/failure rate;
* kernel-pair evaluations;
* generator forward passes;
* setup-inclusive wall time;
* memory use for cross-fitting;
* trajectory spread/covariance and a homogeneity index;
* median and lower quantiles of `P*Q`, `||Delta||`, and ESS.

The last line is essential: it tests whether each intervention repairs its
intended mechanism rather than merely changing the endpoint.

### E4. Pre-registered statistical gate

Use target/initialization cells as the scientific units and seeds as repeated
runs inside a cell. The primary summary is the target-balanced geometric mean
of paired ED2 ratios

```text
ratio = ED2_NCJ / ED2_paper.
```

Use a hierarchical bootstrap that first resamples target cells and then paired
seeds within cells. Row-wise bootstrap is diagnostic only.

The frozen NCJ method passes the particle-level general-improvement gate only
if all of the following hold on the untouched test registry:

1. target-balanced geometric-mean ED2 ratio `<= 0.80`;
2. hierarchical 95% confidence interval upper endpoint `< 1`;
3. at least 60% of target/initialization cells have paired-median ratio `< 1`;
4. no target family has an aggregate median degradation greater than 10%;
5. both Gaussian-mixture and non-Gaussian subgroup confidence intervals have
   upper endpoint `< 1`, or one is explicitly declared an exploratory failure;
6. missing-mode Kaplan--Meier recovery is no worse than the paper baseline;
7. divergence is not higher by more than a pre-declared tolerance;
8. the conclusion survives comparison with the compute-matched baseline.

Criterion 5 is deliberately demanding. If it fails, the result may still be a
useful conditional improvement, but it is not the desired general claim.

Do not change the gate after inspecting test results. Any new policy requires
a new validation/test split.

### E5. Learned-generator transfer

Run this phase only if the frozen NCJ arm passes E4.

1. Implement a small fixed MLP generator for 1-D, 2-D, and selected 5-D
   targets.
2. Use identical architecture, initialization, latent batches, optimizer,
   update count, and data batches across paired arms.
3. Stop-gradient and target-vector semantics must match the paper training
   objective exactly except for the declared NCJ field construction.
4. Compare exact paper, compute-matched paper, and frozen NCJ.
5. Repeat the six-arm ablation on a reduced registry to confirm transfer of
   the same mechanisms.

Generator-level acceptance:

* hierarchical ED2 confidence interval favors NCJ;
* the gain occurs in more than one target family and initialization regime;
* mode coverage and failure rate are non-inferior;
* tuning, compute, and wall cost are included.

Only after E4 and E5 pass may the project state a general low-dimensional
algorithmic improvement.

## 6. Adaptive bandwidth is a later extension

The formal converse says a valid fixed Laplace bandwidth is sufficient for
identifiability; bandwidth is primarily a conditioning and finite-sample dial.
The collapse atlas shows why a coarse scale can overcome missing-mode
metastability, but the recent matched experiment did not support the proposed
geometry bandwidth as a general improvement.

Therefore adaptive bandwidth is deferred until NCJ is understood. If needed,
the next candidate should use only observable field diagnostics:

* disagreement angle between coarse- and fine-scale `Delta`;
* fine-scale ESS or affinity-mass collapse;
* stability of the normalized signal over consecutive batches;
* residual and homogeneity diagnostics.

It must not use true mode count, true component masses, oracle separation, or
held-out target labels. Bandwidth adaptation must be added as a separate
factorial arm on another fresh split.

## 7. Contemporary comparison controls

The scientific comparison is primarily against the original paper estimator,
but current drifting literature already contains nearby corrections. At least
the following should be implemented or cited as secondary controls before a
publication-level novelty claim:

1. **Analytical Bias Correction (ABC)**,
   <https://arxiv.org/abs/2604.27239>: corrects the leading self-normalized
   minibatch-centroid bias.
2. **Sinkhorn Drifting**,
   <https://arxiv.org/abs/2603.12366>: replaces one-sided normalization with
   two-sided balanced couplings and reports improved temperature robustness.
3. **Kernel-Gradient Drifting**,
   <https://arxiv.org/abs/2605.10727>: changes the displacement field into a
   conservative smoothed-score field.

ABC is the most important low-cost comparator. Sinkhorn and kernel-gradient
methods change more of the estimator and may be reported as context if a full
matched implementation is too expensive.

The possible new contribution is not "adding noise," "removing a mask," or
"normalizing a vector" in isolation. It is the audited synthesis:

* remove a formally proved identifiability-inert and exponentially attenuating
  gain;
* eliminate the finite self-pair through cross-fitting;
* use injective symmetric smoothing to break homogeneous particle dynamics;
* retain an end-to-end zero-drift identification guarantee;
* validate the resulting method under a pre-registered generalization gate.

An exact novelty statement requires a dedicated literature review after the
algorithm survives validation.

## 8. Artifact and provenance requirements

Each run directory must contain:

```text
manifest.json
rows.csv or rows.parquet
summary.json
trajectories.npz
final_particles.npz
source_hashes.json
source_snapshots/
stdout.log
```

The manifest must record:

* command and complete configuration;
* Git commit and full dirty status;
* hashes and snapshots of every executed source and frozen protocol;
* Python and package versions;
* machine/platform metadata;
* master seed and derived seed scheme;
* target-registry hash;
* expected and realized row counts;
* all stopping/censoring rules.

Source status must be clean before a scientific run. If an emergency rerun is
made from a dirty tree, the full diff must be stored and the run cannot replace
the clean primary run without explanation.

## 9. Stop rules and claim discipline

Stop or redesign before E5 if any of these occurs:

* constant gain causes broad near-equilibrium instability that global clipping
  cannot fix;
* symmetric jitter improves splitting only by unacceptable target blurring;
* cross-fitting gains disappear under compute matching;
* one family supplies nearly all aggregate improvement;
* validation requires target-specific rules;
* the frozen test fails the general gate.

Allowed conclusions by outcome:

| Outcome | Defensible conclusion |
|---|---|
| E4 and E5 pass | General low-dimensional improvement over the exact paper implementation under matched conditions |
| E4 passes, E5 fails | Empirical-particle estimator improvement that does not transfer through the tested generator |
| Only one mechanism/family improves | Conditional design rule, not a general algorithmic improvement |
| E4 fails | No general improvement; retain mechanism findings and do not run a flagship generator comparison |

No result from this program may be described as outperforming the paper's
ImageNet model unless a later, separately designed image-scale reproduction is
performed.

## 10. Concrete execution order

1. Commit this plan before implementation.
2. Add the reusable NCJ field and invariant tests.
3. Prove/package T1 and T2; run the full trust audit.
4. Package the cross-fitted statistical statement T3.
5. Create and commit the frozen validation/test protocol and target hashes.
6. Run smoke tests only for software correctness.
7. Run E1 validation and freeze `sigma/tau`, norm clip, and global step.
8. Commit the frozen winner before touching the test registry.
9. Run the full untouched E4 test and write results without changing the gate.
10. If and only if E4 passes, implement and run E5.
11. Add T4 and, if supported by data, T5 as the explanatory theorem layer.
12. Compare with ABC and report Sinkhorn/kernel-gradient context.
13. Update `ResearchStatus.md`, `DynamicsRoadmap.md`, and the presentation with
    the exact level of claim earned.

The first practical implementation milestone is complete when steps 1--5 are
committed, the exact-paper regression tests pass, T1--T3 are trust-audited, and
the validation/test registries are frozen. No performance claim is made at
that milestone.
