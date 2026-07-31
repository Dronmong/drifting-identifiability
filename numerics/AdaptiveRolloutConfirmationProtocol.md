# Dimension-Adaptive Rollout Confirmation Protocol

## Status

This protocol was frozen before generating or evaluating the confirmation
registry. The consumed development artifacts used to choose the candidate are:

- `conditioned_transport_runs/20260723-161008-consumed/` (mechanism screen);
- `conditioned_transport_runs/20260723-161547-consumed/` (selected-rule
  reconstruction).

No confirmation target is contained in either development artifact.

## Candidate and causal hypothesis

The candidate is `cta-exact-adaptive-rollout`.

- dimensions 2 and 4: one PSQT teacher step, exactly the prior fixed control;
- dimension 8: two same-particle PSQT substeps with rank recomputation;
- dimension 16: four such substeps;
- step size: `0.5` at every substep;
- post-rollout local field weight: `0.25`;
- active direction count: `32`;
- local support: fixed `M=128` below 8D and target-only
  variance-per-node `M=256` at and above 8D.

The rule is dimension-only. It does not inspect candidate outcomes, target
component labels, or evaluation metrics. The hypothesis is that sparse
high-dimensional occupancy requires persistent movement of the same free
particles, whereas repeating the transport in low dimension over-transports
already well-resolved targets.

The comparators are:

1. `cta-exact-fixed-control`, the prior trusted one-step implementation;
2. `cta-exact-gated-hybrid`, the previous representative-capacity repair; and
3. `paper-neural-optimized`, the matched implementation of the paper arm.

The experiment does not claim lower compute. It tests whether a measured,
moderate increase in projection/sort/kernel work buys robust distributional
quality and rare-component occupancy.

## Frozen unseen design

- registry master seed: `20261107`;
- two instances in every dimension/family cell;
- dimensions: 2, 4, 8, 16;
- families: balanced GMM, rare GMM, correlated Student-t, nonlinear;
- 32 independent target instances total;
- both concentrated and broad initializations;
- consumed profile: 20,480 generator examples and 20,480 target examples;
- 2,048 output samples;
- 64 registered training directions and 128 held-out directions;
- randomized arm order within each target;
- two dense representative-field audit calls per local arm.

The independent statistical unit is a target instance. The two
initializations are reduced within target by a geometric mean before
uncertainty is computed.

## Primary endpoints and gates

For candidate versus fixed control:

1. paired geometric mean ED2 ratio below one;
2. paired geometric mean held-out SW1 ratio below one;
3. stratified target bootstrap 95% upper bound below one for both metrics.

The bootstrap resamples the two target instances independently inside each of
the 16 dimension/family strata, then averages log ratios equally over strata.

Candidate versus the paper arm is a confirmatory benchmark with the same two
quality gates. Candidate versus gated hybrid is secondary and is reported
with the same statistics.

## Mechanism and safety endpoints

The report must also include:

- ED2 and SW1 ratios separately by dimension and family;
- rare-component target-calibrated core mass and bridge mass;
- rare mass error;
- whether the rollout teacher ever reaches and finishes in the rare core;
- whether rare-core teacher particles survive neural amortization;
- training quantile RMSE;
- wall time, projection scalar products, sort work, and kernel-pair ledger;
- low-dimensional identity with the fixed control;
- representative approximation diagnostics and the deep artifact audit.

Target component parameters may be used only for post-hoc diagnostics.
Training, routing, rollout selection, and stopping remain label-free.

## Interpretation rules

- Passing the primary gates supports a scoped synthetic proof of concept, not
  an ImageNet or universal superiority claim.
- A quality win with higher compute is a quality/compute tradeoff, not an
  efficiency win.
- Improved bridge occupancy without core occupancy is not rare-mode recovery.
- A teacher core hit without generator core retention diagnoses amortization;
  no teacher core hit diagnoses the transport horizon.
- No parameter, threshold, target, endpoint, or arm may be changed after the
  first confirmation run begins.

