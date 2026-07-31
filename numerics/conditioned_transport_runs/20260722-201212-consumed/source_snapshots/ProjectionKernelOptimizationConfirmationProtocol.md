# Projection/kernel optimization confirmation protocol

## Status and freeze

This protocol was written before evaluating the fresh confirmation registry.
It separates the post-selected development evidence in
`ProjectionKernelCostOptimizationResults.md` from the confirmation below.

The frozen target registry is generated with:

```powershell
uv run --with numpy python numerics/generate_neural_pooled_rank_registry.py `
  --output numerics/projection_kernel_confirmation_registry.json `
  --master-seed 20260817
```

The registry contains one newly generated target for each of the 16
dimension/family cells.  Its parameters, train/held-out directions, target
samples, model initialization, latent samples, minibatch order, and bootstrap
seeds are fixed before any confirmation endpoint is inspected.

## Arms and budgets

Every artifact includes the current optimized paper-neural implementation in
the same process and randomized execution order as the conditioned-transport
arms.  The comparison therefore does not reuse historical wall-clock
measurements or the legacy two-softmax field.

All arms receive 20,480 generator-example evaluations.  The paper comparator
uses batches of 64 and the current exact normalized Algorithm-2 field with
`tau = 1` and the registered gain.  Conditioned transport uses 20 macro-steps
of 512 particles, followed by 20,480 student generator evaluations.

The frozen primary arm is:

- exact target atlas;
- conditioned quadratic direction bank;
- 32 active directions per macro-step, selected in complete orthogonal blocks
  with balanced exposure;
- all 20 local-field calls; and
- 128 weighted projection-tree representatives per positive and negative
  support.

Two ablations are run without changing any other choice:

- dense local supports (`M = 512`); and
- aggressive compression (`M = 64`).

The KLL arms emitted by the runner are secondary diagnostics.  The primary
claim uses the deterministic exact-atlas hybrid only.

## Outcomes

The two co-primary quality outcomes are:

1. energy distance squared (ED2); and
2. sliced Wasserstein-1 on 128 held-out directions (SW1).

Both are paired by frozen target.  Lower is better.  We report geometric mean
ratios with a target-level percentile-bootstrap 95% interval, paired median
ratios, and win counts.  The bootstrap seed is `2026081709` and the number of
resamples is 20,000.

The primary quality result is considered successful only if, for both ED2 and
SW1:

- the geometric mean ratio of the primary arm to the matched paper comparator
  is below one;
- the upper endpoint of its 95% bootstrap interval is below one; and
- the primary arm wins on at least 12 of 16 targets.

This is a synthetic low-dimensional confirmation, not an ImageNet or
state-of-the-art generative-model claim.

## Efficiency and support outcomes

Efficiency is reported in distinct scopes:

- online training wall time;
- conditioned-direction plus atlas setup plus online training wall time;
- exact kernel-pair ledger;
- target-example access ledger;
- projection and sorting ledgers; and
- serialized atlas/KLL storage.

Wall time is implementation-, hardware-, and execution-order-dependent.  The
matched comparator, deterministic randomized arm order, and three separate
launches reduce but do not eliminate timing noise.  No peak-memory advantage
will be claimed unless separately measured.

For every target we also report mode coverage.  On rare-GMM targets we report
rare-component mass error.  The balanced projection tree guarantees equal
leaf populations (four observations per leaf at `M = 128` and eight at
`M = 64`); it does **not** use component labels and does not guarantee that a
semantic rare mode receives a dedicated representative.  Rare-mode safety is
therefore an empirical outcome, not an algorithmic premise.

## Reproducibility gates

Before fitting any target, the runner must pass:

- exact projection/kernel refactor equivalence;
- the weighted-representative full-support field and mass identities;
- the actual runner's `M = B` field and one-update parameter equivalence;
- direction scheduling, quadratic-frame, and transport-step regressions.

Each artifact must contain source snapshots, copied registry and sidecar,
ledger rows, saved outputs and references, serialized KLL states, hashes of
all result payloads, Git commit/dirty state, and a passing deep audit that
recomputes ED2 and held-out SW1 from saved arrays.

Invalid smoke artifacts and the earlier bandwidth-confounded K2 artifacts are
not evidence for this protocol.

