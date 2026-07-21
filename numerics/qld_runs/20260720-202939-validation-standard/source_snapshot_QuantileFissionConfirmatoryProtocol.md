# QLD confirmatory protocol (frozen before execution)

**Protocol ID:** `QLD-confirmatory-v1`  
**Registry SHA-256:** `7FEF49789904464A3103E09A166AD1C49A28C72DCAE00C61B87D72AA0CB1B8F8`  
**Status:** pre-registered locally before validation or test execution.

## Claim being tested

Quantile-to-Laplace Drifting (QLD) improves final distributional error over the
paper's exact Algorithm 2 on one-dimensional fission problems under missing-mode
or concentrated learned-generator initialization.

This is not a claim about arbitrary initialization, dimensions above one, real
image features, or ImageNet FID. Broad and far starts are outside the primary
gate; the already-observed far-start failure remains documented.

## Frozen QLD arm

- repository `TanhMLP` and Adam implementation;
- 1,200 updates, batch size 128;
- updates 1--840: exact minibatch rank coupling
  `V_i = y_(rank(x_i)) - x_i`;
- updates 841--1,200: exact paper Algorithm 2;
- refinement bandwidth `tau = 0.5`;
- paper eye mask enabled during refinement;
- no mode labels, target weights, component widths, or target parameters are
  read by the training algorithm.

No QLD hyperparameter may be selected or changed using validation or test.

## Frozen paper baseline selection

The validation split consists of the six `validation_targets` in the sealed
registry, evaluated under both primary initializations and eight paired seeds.
For each `tau` in `{0.2, 0.5, 1.0, 2.0, 4.0}`, train exact paper Algorithm 2 for
the same 1,200 updates and batch size. Select the single `tau` minimizing the
geometric mean of cell-median ED-squared. Freeze that one bandwidth globally
before opening the test results.

The test run additionally evaluates all five paper bandwidths. An
oracle-per-target diagnostic may choose the best paper bandwidth on each test
target, jointly across its two initializations. This diagnostic is intentionally
advantaged and cannot determine the primary gate.

## Sealed test suite

The registry contains 16 targets not used in QLD development or validation:

- eight equal-weight mixtures (`K=7` through `K=32`);
- three unequal-weight mixtures;
- two heteroscedastic mixtures;
- one overlapping mixture;
- one remotely contaminated mixture;
- one connected heavy-tailed Student target.

Primary initializations are `missing` and `concentrated`. Each target/init/arm
cell uses 20 paired seeds. Pairing fixes initial model parameters, latent
batches, target batches, and final evaluation references across arms.

## Metrics

Primary cell error is the median final ED-squared over 20 seeds. The primary
aggregate is the geometric mean of `QLD median / selected-paper median` over all
32 target/initialization cells.

The confidence interval is a paired hierarchical target bootstrap: resample
the 16 targets, then resample paired seeds within both initialization cells,
recompute cell medians, and aggregate geometrically. Use 10,000 replicates.

Secondary metrics are one-dimensional sliced-W1, weighted mode reach,
nearest-component mass L1, time to 90% weighted reach, divergence, wall time,
kernel-pair count, and sorting-work proxy. Oracle component information is used
only for diagnostics, never by either training arm.

## Gate fixed before results

QLD passes only if all conditions hold:

1. target-balanced primary ED2 ratio is at most `0.80`;
2. paired target-bootstrap 95% upper confidence bound is below `1.0`;
3. QLD wins ED2 in at least 60% of primary cells;
4. no predefined primary target family has geometric ED2 ratio above `1.10`;
5. QLD has no more divergent runs than the selected paper arm.

Secondary metrics cannot rescue a failed primary gate. The oracle-per-target
comparison is reported regardless of outcome.

## Reproduction

```powershell
uv run --with numpy --with scipy python numerics/run_qld_confirmatory.py --stage validation --profile standard
uv run --with numpy --with scipy python numerics/run_qld_confirmatory.py --stage test --profile standard --selection <validation-selection.json>
```

Run directories, manifests, row-level CSV files, selections, source snapshots,
and summaries are written below `numerics/qld_runs/`.
