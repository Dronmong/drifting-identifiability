# Low-dimensional performance roadmap

*Proposed next program after the audited Phase C validation. The Phase C
checkpoint is commit `627d23c`. This document separates the immediate
performance question from the longer quantitative-theory program in
`DriftingIdentifiability/DynamicsRoadmap.md`.*

## Executive decision

The next priority should be a **matched low-dimensional performance study**,
not an immediate attempt to complete all of Phase B.

The project does not presently need an ImageNet-scale reproduction. It does
need to answer a narrower and attainable question:

> Under identical low-dimensional data, initialization, estimator, compute,
> and model architecture, do the proposed bandwidth, step-size, and masking
> modifications outperform a strong implementation of the original drifting
> procedure?

Phase B remains valuable, especially B1, but it cannot by itself demonstrate
that the modified procedure performs better. The modification should first be
made precise and tested with the paper estimator. Once a winning procedure is
frozen, B1 can explain why it works without risking a long proof about an
algorithm that later changes.

## What is already established

The formal and numerical programs have reached different but complementary
levels.

### Formal level

The project has a machine-checked population converse for the Euclidean
Laplace kernel: vanishing population drift identifies the two probability
measures in the formalized setting. It also has certified results about
fission instability and finite-range mass blindness.

These results establish that the Laplace field contains enough information and
identify structural mechanisms in its dynamics. They do not supply a global
finite-time convergence rate or establish the performance of a trained model.

### Synthetic particle level

The corrected Phase C experiments establish that:

* a bandwidth much smaller than inter-mode separation creates a severe
  missing-mode transport barrier;
* coarse bandwidth repairs that barrier, although fixed coarse versus
  coarse-to-fine scheduling is geometry-dependent;
* the full coupled generator provides a correct local Euler ceiling, but that
  ceiling is not itself a good default training step;
* masking is harmful at very small particle counts under both SNIS and the
  paper's bi-softmax affinity;
* no endpoint in the corrected study passed the strengthened stable-wrong-
  equilibrium test.

### What is not yet established

The present experiments do not yet show that one final modified algorithm
consistently beats the base procedure because:

1. Phase C1 and C2 primarily use the simpler SNIS field. The exact paper
   bi-softmax estimator was used in C3, but not throughout the bandwidth and
   step-size comparisons.
2. The current fixed-time annealing rule is not universally superior to fixed
   coarse bandwidth.
3. Independently updated particles do not test whether the improvement
   survives coupling through shared neural-network parameters.
4. Some Phase C policies use clean-mixture geometry estimates. A final method
   must separate information available to the algorithm from oracle labels
   used only for evaluation.

The strongest honest current wording is therefore:

> The formal analysis generated useful design mechanisms, and corrected
> synthetic experiments validate those mechanisms, but a matched
> low-dimensional base-versus-modified-model comparison remains to be done.

## Scope of the next benchmark

The study has two successive tracks.

### Track P: empirical particle model

This track is the cheapest controlled test. The model law is an empirical
particle cloud. Every comparison uses the exact row/column bi-softmax
Algorithm-2 estimator, the same target minibatches, the same initialization,
and the same kernel-evaluation budget.

Track P answers whether the proposed changes improve the estimator-level
dynamics when the particles themselves are the trainable state.

### Track G: small learned generator

This track uses a small shared generator, for example

```text
z in R^2  ->  G_theta(z) in R or R^2,
```

with a fixed MLP architecture. The base and modified methods must share the
same architecture, initialization distribution, optimizer, minibatches, and
compute budget. Only the drifting policies may differ.

Track G is the decisive transfer test: it determines whether a particle-level
gain survives parameter sharing and gradient-based training without requiring
high-dimensional images or a large research group.

## Experimental principles

Every result in this phase must follow these rules.

1. **Use a strong baseline.** Tune the original fixed-bandwidth/fixed-step
   procedure on validation problems rather than comparing against a weak
   arbitrary default.
2. **Change one factor at a time.** Report bandwidth-only, step-only,
   mask-only, and combined arms.
3. **Use the same estimator.** The base and modified arms must both use the
   exact paper bi-softmax estimator unless an experiment is explicitly labeled
   as an SNIS diagnostic.
4. **Match compute.** Equalize kernel-pair evaluations and report wall time.
   An adaptive method may not silently receive more field evaluations.
5. **Prevent oracle leakage.** Mode labels, true separation, and true component
   width may be used for diagnostics but not by the final policy. Geometry
   supplied to the policy must be estimated from target samples alone.
6. **Separate selection from evaluation.** Develop on training targets, choose
   hyperparameters and triggers on validation targets, and report the final
   comparison once on held-out target geometries.
7. **Use paired randomness.** Each arm receives the same initialization and
   target-batch stream for a seed. Report paired uncertainty rather than only
   independent medians.
8. **Retain complete provenance.** Save source hash, commit, dirty state,
   configuration, command line, versions, per-seed rows, trajectories,
   evaluation cost, and wall time.

## Concrete implementation roadmap

### D0 — freeze and verify the low-dimensional baseline

Goal: make “the base model” precise before attempting to beat it.

1. Extract the dimension-general paper bi-softmax field already cross-checked
   in `driftbench_v2.py` into a reusable benchmark module.
2. Define one particle baseline and one learned-generator baseline from the
   same estimator and the paper-style fixed temperature/mask policy.
3. Write invariant tests for translation equivariance, finite outputs,
   matched-positive/negative cancellation, and agreement with
   `driftlab.compute_v_paper` in one dimension.
4. Tune the baseline's fixed temperature and step size on validation targets
   under the same budget later granted to the modified method.
5. Freeze the chosen baseline configuration before evaluating held-out
   targets.

Acceptance gate D0:

* independent implementations agree numerically;
* the baseline has no known transcription or singleton-Jacobian bug;
* its tuning cost and final configuration are recorded;
* no test target has been used for selection.

### D1 — port the Phase C mechanisms to the exact estimator

Goal: determine which Phase C findings transfer from SNIS to Algorithm 2.

Run the following exact-estimator ablation arms:

1. tuned fixed baseline;
2. coarse bandwidth only, with the baseline step held fixed;
3. coarse-to-fine bandwidth only, with equalized field-evaluation cost;
4. generator-informed safety step only;
5. particle-count-aware masking only;
6. all three modifications combined.

The C2 generator must differentiate the complete coupled bi-softmax particle
field. The spectral boundary is used only as a ceiling. Candidate operating
rules should include conservative fractions such as `0.05 eta*`, `0.1 eta*`,
and `0.25 eta*`, together with inexpensive fixed multiples of `tau`.

Acceptance gate D1:

* every claimed mechanism has been tested using the exact estimator;
* bandwidth and step-size effects remain causally separated;
* at least one combined candidate clearly improves the validation aggregate;
* otherwise the project records that the SNIS improvement did not transfer
  and redesigns the candidate before further theory.

### D2 — replace fixed-time annealing with an adaptive policy

Goal: produce one precise modified algorithm rather than a collection of
post-hoc rules.

The current “anneal after 70% of updates” policy should be treated as a
baseline schedule, not the final proposal. Candidate triggers must be
observable without true mode labels. The validation study should compare:

* a plateau in the coarse-field residual or held-out sample discrepancy;
* agreement in direction between coarse- and fine-bandwidth drift estimates;
* stabilization of data-estimated target scales across minibatches;
* an affinity-connectivity or effective-sample-size threshold.

Mode coverage and true component masses may evaluate a trigger on synthetic
data, but they may not trigger the final algorithm.

The selected policy must specify completely:

```text
initial bandwidth estimate;
refinement trigger;
bandwidth update rule;
step-size safety rule;
mask rule;
fallback behavior when diagnostics are ambiguous.
```

Acceptance gate D2:

* select exactly one combined policy using validation targets only;
* demonstrate that its gain is not solely caused by extra evaluations;
* reject it if it simply overfits Gaussian-mixture geometry.

### D3 — held-out empirical-particle benchmark

Goal: decide whether the frozen modified procedure outperforms the frozen base
procedure in the intended low-dimensional setting.

Recommended target families:

* balanced and unequal-weight 1-D Gaussian mixtures;
* separated and nearly overlapping 2-D Gaussian mixtures;
* heteroscedastic mixtures;
* rings or concentric circles;
* two moons;
* one connected skewed or heavy-tailed target;
* both missing-mode and already-covered initializations.

The held-out axes should include separation, component width, number of modes,
particle count, target minibatch size, and unequal target masses. Use at least
20 paired seeds for the final aggregate unless a power analysis justifies
fewer.

Primary metrics:

* final squared energy distance;
* time to a predeclared energy-distance threshold with right censoring;
* kernel-evaluation and wall-clock cost.

Secondary metrics:

* sliced Wasserstein distance;
* mode coverage and mass error where a true modal decomposition exists;
* on-support drift residual;
* divergence/failure frequency.

Acceptance gate D3:

The modified particle procedure advances only if, on held-out targets:

1. its paired aggregate median error improves by at least 20%;
2. the upper endpoint of a paired 95% bootstrap confidence interval for the
   aggregate error ratio is below `1`;
3. it does not degrade more than 20% of target cells by over 10%;
4. it preserves or improves threshold time under an equal compute budget;
5. the conclusion is not dependent on one Gaussian-mixture family.

If it fails this gate, return to D2. Do not proceed by weakening the success
criterion after observing the test results.

### D4 — learned-generator transfer

Goal: test the same frozen policies with shared parameters.

1. Implement a small fixed MLP generator for 1-D and 2-D outputs.
2. Transcribe the paper's low-dimensional training update faithfully and add
   numerical invariants for its loss/gradient direction.
3. Use identical model initialization, latent batches, target batches,
   optimizer, update count, and compute accounting across paired arms.
4. Compare the frozen D0 baseline with the frozen D2 modification. Do not
   retune the modified method on held-out targets.
5. Repeat the main ablations to identify whether any gain comes from bandwidth,
   step size, masking, or an interaction among them.

Acceptance gate D4:

* the combined method improves the held-out learned-generator aggregate with a
  paired confidence interval favoring it;
* the gain occurs on more than one target family and initialization regime;
* failures, wall time, and tuning costs are included;
* any claim is phrased as improvement over the **low-dimensional base
  implementation**, not over the paper's reported image benchmarks.

### D5 — consolidate the result

If D3 and D4 pass, create a concise result package containing:

* the complete algorithm specification;
* base-versus-modified tables and paired confidence intervals;
* ablations and failure cases;
* raw manifests and trajectories;
* an explicit scope statement;
* a link from `DynamicsRoadmap.md` and `ResearchStatus.md`.

At that point the defensible performance claim becomes:

> On held-out low-dimensional distributions and under matched estimator,
> architecture, and compute conditions, the theoretically motivated modified
> drifting procedure outperforms the tuned base implementation.

If only D3 passes, report an estimator-level particle improvement and state
that transfer through a learned generator failed or remains open.

## When to return to Phase B

Phase B should begin after D2 freezes the candidate algorithm, and preferably
after D3 confirms a particle-level gain. It need not wait for every learned-
generator run, so B1 and D4 can proceed in parallel once the algorithm stops
changing.

### First: B1 residual floor

B1 is the highest-value theoretical target. A bound of the form

```text
0 < ||V|| <= C exp(-L/tau)
```

on imbalanced separated clusters would explain both sides of the numerical
phenomenon: the wrong-mass state is not an equilibrium, but fine-bandwidth
transport can be exponentially slow. It would provide a principled reason to
start at a separation-scale bandwidth.

The theory should be written for the policy or target family that survives D3,
not for every candidate tried during D1--D2.

### Second: B2 metastability time

Attempt two-sided relaxation-time bounds only after B1 clarifies the constants
and relevant state variables. B2 is scientifically valuable, but it is more
difficult and less directly necessary for the low-dimensional performance
claim.

### Reframe B3 mask theory

The old B3 wording presupposes a stable wrong masked equilibrium. The corrected
Phase C study did not reproduce such a state under its strengthened gate.
Before formal work, restate B3 around claims actually supported by the data:

* finite-`N` masked-versus-unmasked bias;
* dependence on particles per mode and target batch size;
* a residual or law-level perturbation bound;
* sufficient conditions for masking to help or hurt local contraction.

A stable-wrong-equilibrium theorem should be attempted only if a fully
reproducible corrected example survives long continuation, residual,
spectrum, and perturb-and-return checks.

## Recommended concrete order

1. **Checkpoint Phase C.** Completed at commit `627d23c`.
2. **Write and verify the exact-estimator baseline (D0).**
3. **Port C1/C2 and all ablations to the paper estimator (D1).**
4. **Select a non-oracle adaptive bandwidth policy (D2).**
5. **Run the held-out empirical-particle benchmark (D3).**
6. **If D3 passes, freeze the algorithm and start B1.**
7. **In parallel, transfer the frozen algorithm to the small MLP generator
   (D4).**
8. **Consolidate the performance result (D5).**
9. **Proceed to B2 only if B1 succeeds; rewrite B3 before attempting it.**

This order minimizes wasted theory work, produces the earliest credible
performance comparison, and preserves a clean separation between empirical
improvement, mathematical explanation, and large-scale future validation.

## Proposed files and artifacts

The implementation phase should use a fresh namespace rather than modifying
the historical Phase C runs in place:

```text
numerics/lowdim_drift.py                 reusable exact-estimator dynamics
numerics/lowdim_benchmark.py             D0--D3 command-line runner
numerics/lowdim_generator.py             D4 learned-generator runner
numerics/LowDimPerformanceProtocol.md    frozen experiment specification
numerics/LowDimPerformanceResults.md     final tables and conclusions
numerics/lowdim_runs/<run-id>/           manifests, rows, trajectories
numerics/lowdim_figures/                  uncertainty and ablation figures
```

Do not overwrite the Phase C artifacts. They are the audit trail that led to
this next program.
