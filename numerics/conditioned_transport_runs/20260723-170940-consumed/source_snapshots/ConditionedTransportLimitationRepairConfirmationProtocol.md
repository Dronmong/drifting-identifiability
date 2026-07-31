# Conditioned transport limitation-repair confirmation protocol

**Status:** frozen before target generation or endpoint inspection  
**Date:** 2026-07-23  
**Development plan:** `ConditionedTransportLimitationRepairPlan.md`

## 1. Claim under test

The experiment tests one selected repair only:

> Replacing the historical seven-fixed-direction `M=128` projection tree with
> a per-node maximum-variance registered-direction tree, using `M=128` below
> dimension 8 and `M=256` at dimensions 8 and 16, reduces representative-field
> error and improves final neural distribution quality without losing the
> already confirmed advantage over the repository paper port.

The cross-fitted local controller is not included. It failed its consumed-
registry development gate. Radius-priority and tail-reserve trees are also
excluded because they were slower and had worse lower-tail field cosine.

## 2. Frozen registry

Generate a new registry with:

```powershell
uv run --with numpy python numerics/generate_neural_pooled_rank_registry.py `
  --output numerics/conditioned_transport_repair_confirmation_registry.json `
  --master-seed 20260923 `
  --variants-per-cell 2
```

This produces:

- dimensions `2`, `4`, `8`, and `16`;
- balanced GMM, rare GMM, correlated-t, and nonlinear families;
- two independently generated targets per dimension/family cell; and
- 32 target instances in total.

The registry and its SHA-256 sidecar must exist before either model run.

## 3. Frozen arms

Every target is evaluated from both `concentrated` and `broad` generator
initializations.

### 3.1 Repaired candidate

- arm: `cta-exact-hybrid`;
- exact persistent target atlas;
- active directions: `32`;
- local field on all 20 macro-steps;
- representative strategy: `variance-per-node`;
- `M=128` in dimensions 2 and 4;
- `M=256` in dimensions 8 and 16;
- capacity threshold: dimension 8;
- local weight: fixed `0.25`;
- two dense field-audit calls per target/initialization; and
- all other optimizer/teacher settings unchanged.

### 3.2 Current-model control

- arm: `cta-exact-fixed-control`;
- same global transport, direction schedule, local weight, optimizer, target
  batches, latent samples, and evaluation data;
- historical `fixed-level` representative tree; and
- `M=128` in every dimension.

### 3.3 Paper comparator

- arm: `paper-neural-optimized`;
- current exact normalized implementation;
- same target and generator-example budget; and
- execution in the same process and randomized target-specific arm order.

## 4. Frozen runs

Run the same registry twice:

```powershell
# concentrated
uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy `
  --with datasketches==5.2.0 `
  python numerics/run_conditioned_transport_development.py `
  --profile consumed `
  --initialization concentrated `
  --registry numerics/conditioned_transport_repair_confirmation_registry.json `
  --arms cta-exact-hybrid cta-exact-fixed-control paper-neural-optimized `
  --active-directions 32 `
  --local-representatives 128 `
  --high-dimensional-local-representatives 256 `
  --representative-capacity-threshold-dimension 8 `
  --representative-strategy variance-per-node `
  --representative-audit-calls 2

# broad: identical except --initialization broad
```

No target, seed, capacity, direction, weight, tolerance, or endpoint may be
changed after the first run begins.

## 5. Analysis unit and uncertainty

The independent unit is a target instance, not a target/initialization row.

For each target and metric:

1. compute the repaired/control ratio separately for concentrated and broad
   initialization;
2. take their geometric mean; and
3. bootstrap the resulting 32 target-level log ratios.

Use 20,000 percentile-bootstrap draws with seed `2026092307`, stratified over
the 16 dimension/family cells so both target variants remain represented.

## 6. Outcomes

### 6.1 Co-primary quality endpoints

- energy distance squared (`ED2`);
- held-out sliced Wasserstein-1 (`SW1`).

### 6.2 Mechanism endpoints

- representative-field relative-L2 error;
- minimum representative-field cosine;
- row- and column-mass relative-L2 error;
- positive/negative RMS radius;
- positive/negative maximum radius; and
- partition projection/sort work.

### 6.3 Support endpoints

The exploratory support definitions developed after the previous confirmation
are now frozen secondary endpoints:

- target-calibrated support precision;
- target-calibrated support recall;
- off-support occupancy;
- normalized mean error;
- relative covariance Frobenius error;
- mixture-mode coverage; and
- rare-component mass error.

The target support radius uses the 95th percentile of target-evaluation
5-nearest-neighbor distances to a disjoint target calibration half.

### 6.4 Cost endpoints

- generator-example evaluations;
- exact kernel pairs;
- target accesses;
- target-atlas projections;
- training projections;
- partition projections;
- sorting ledger;
- online CPU wall time; and
- setup-plus-training CPU wall time.

The two dense diagnostic calls are excluded from the algorithmic kernel-pair
ledger but remain inside measured wall time.

## 7. Frozen gates

### Gate A: quality improvement over the current model

For both ED2 and held-out SW1:

- repaired/control geometric-mean ratio `< 1`;
- upper endpoint of the target-bootstrap 95% interval `< 1`; and
- repaired wins on at least `20/32` target-reduced comparisons.

This is deliberately stronger than demonstrating another win over paper.

### Gate B: mechanism improvement

Across dimensions 8 and 16:

- geometric-mean positive field-error ratio repaired/control `<= 0.75`;
- minimum field cosine is no lower than control minus `0.01`;
- median row-mass relative error does not increase by more than `10%`; and
- median column-mass relative error does not increase by more than `10%`.

### Gate C: retain the paper advantage

For both ED2 and held-out SW1:

- repaired/paper geometric-mean ratio `< 1`;
- upper endpoint of the 95% interval `< 1`; and
- repaired wins on at least `24/32` targets.

### Gate D: support non-inferiority

Relative to the current-model control:

- median support precision decreases by no more than `0.02`;
- median support recall decreases by no more than `0.01`;
- minimum mixture-mode coverage does not decrease; and
- median rare-mass error does not increase by more than `20%`.

### Decision

The repaired representative system is promoted only if Gates A through D all
pass. Cost is reported rather than thresholded because the experiment includes
dense diagnostic calls and is CPU-specific.

If Gate B passes but Gate A fails, the conclusion is that representative-field
accuracy is not a sufficiently strong endpoint bottleneck; retain the current
tree.

## 8. Guardrails

1. Do not use the two new target variants for further hyperparameter selection.
2. Do not inspect one initialization before running the other.
3. Do not change the local weight or introduce the failed controller.
4. Do not use evaluation directions for training or capacity selection.
5. Do not use mixture labels in representative construction.
6. Do not call target-reduced bootstrap intervals training-seed uncertainty;
   they quantify heterogeneity across these 32 target instances.
7. Do not claim image-scale, GPU, or universal superiority from this study.
