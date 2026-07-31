# Encoder-independent F1 K=200 confirmation protocol

**Status: FROZEN DESIGN.** This is a fresh confirmation of the narrow F1
question after the audit of the exploratory K=20,000 run. It does not overwrite
or reinterpret `f1.json`.

## 1. Question and scope

For the frozen raw-geometry, smooth-Laplace, RMS-normalized, R11-corrected
free-particle update, can the deployed `random_generator` start acquire
nontrivial CIFAR-10 manifold recall before the same update invalidates a
real-data positive control?

The terminal checkpoint is **K=200**. This is the last previously measured
checkpoint at which every real-data control had recall above 0.5. It is frozen
before the new seeds are generated or run. No checkpoint after K=200 is part of
this confirmation.

This experiment tests the implemented finite-sample update in raw pixel
geometry. It is not the paper's full model, a population-field theorem, an
encoder comparison, or evidence for a global dynamical attractor.

## 2. Frozen configuration

- dataset: CIFAR-10, 32 x 32;
- particles: 512;
- positives per update: 64;
- replay bank: 4,096 unique images per unit;
- checkpoints: `{0, 10, 20, 40, 100, 200}`;
- step size: 0.5;
- geometry: raw pixels;
- kernel: `smooth_laplace`, target ESS fraction 0.05;
- direction: `paper`;
- drift normalization: `rms`;
- denominator floor: `1e-30`;
- self mask: false;
- correction: R11 scalar `corrected_teacher`;
- primary recall gate: `RECALL_GATE = 0.05`;
- positive-control floor: `CONTROL_RECALL_FLOOR = 0.5`;
- units: three new logical units with unit IDs `{100, 101, 102}`;
- regimes: replay and stochastic, reported separately.

Only two arms run:

1. `real_data`, the paired validity control;
2. `random_generator`, the primary F1 arm.

The diagnostic starts from the exploratory study are deliberately omitted.
They do not enter the F1 gate and would only increase cost.

## 3. Allocation and pairing

A single permutation of this project's 40,000-image CIFAR training pool
(the separate 10,000-image pool is reserved for evaluation), generated from
the new confirmation seed namespace, allocates:

- one shared 256-image kernel-calibration subset;
- one unique 4,096-image replay bank per unit;
- one unique 512-image real-data control per unit.

All of those sets are mutually disjoint across all three units. The stochastic
teacher pool is the training pool with the shared calibration set and all three
real-data control sets removed. Replay and stochastic rollouts within a unit
start from the exact same tensor. Primary and control arms within a unit use the
same replay schedule or stochastic minibatch stream.

The kernel is calibrated once on the shared calibration set and the exact same
kernel object is used in all units and regimes.

## 4. Preflight and immutable compatibility

Before any confirmation arm runs:

1. create `f1_k200_freeze.json`, hashing this protocol and every Python source
   file in `numerics/encoder_independent_drifting`;
2. rerun the null-recall calibration into
   `f1_k200_calibration.json` and require `p_null_upper < 0.025` at the fixed
   0.05 gate;
3. rerun the historical/equivalence/reproducibility checks into
   `f1_k200_checks.json` and require every check to pass;
4. rerun the replay-bank veto calibration into `f1_k200_vetoes.json` and
   require at least one valid calibrated veto;
5. calibrate the stochastic full-teacher-pool nearest-training-image veto into
   `f1_k200_stochastic_veto.json` using only held-out evaluation images and
   known exact training copies;
6. verify every artifact SHA-256 sidecar and require its recorded source hashes
   to match the freeze manifest exactly.

The confirmation runner refuses to run if any condition fails. Merely finding
an artifact file is not sufficient.

## 5. Gate and vetoes

At K=200, for each unit and regime:

```text
replicate_pass :=
    random_generator_recall > 0.05
    and all_applicable_calibrated_vetoes_pass
    and paired_real_data_recall > 0.5
```

Replay applies calibrated nearest-bank distance, distinct claimed bank images,
effective rank, one-minus-duplicate-rate, and nearest-neighbour diversity.

Stochastic applies nearest distance to the eligible training pool, effective
rank, one-minus-duplicate-rate, and nearest-neighbour diversity. The finite
replay-bank distinct-count threshold is not reused for stochastic sampling.

Every individual comparison, its threshold, and its Boolean result is stored.
There is no placeholder `vetoes_ok = True` path.

Each regime is adjudicated separately:

- `PASS`: at least two of three valid units pass the full conjunction;
- `FAIL`: fewer than two pass;
- `VOID`: at least two paired real-data controls fail in that regime.

An invalid control is never counted as a failed primary replicate.

## 6. Metrics and uncertainty

Recall is the frozen improved precision/recall statistic in Inception features,
against the same fixed 2,048-image evaluation reference used by calibration.
The gate uses its point value, as calibrated.

The previous particle-with-replacement bootstrap is not used: duplicating
particles changes the generated k-nearest-neighbour radii and did not produce a
valid interval around the observed statistic. A two-sided exact binomial
interval over the 2,048 covered reference indicators is reported only as a
**reference-side diagnostic**; it is not described as total recall uncertainty
and does not enter the gate.

A disjoint 512-image real-vs-real baseline is scored at the identical sample
count. KID, FID, precision, effective rank, spectral and kernel-health metrics
are diagnostics only.

## 7. Declared interpretation

- `PASS` supports the narrow claim that the deployed start enters a
  coverage-bearing state within the verified control-validity window.
- `FAIL` supports the narrow claim that no such coverage was detected at K=200
  in the three new units. It selects the prescribed-bridge/new-objective branch.
- `VOID` means the implemented update did not preserve its positive control for
  the frozen horizon; it does not adjudicate reachability.

The exploratory K=20,000 study may be cited separately as evidence that all
tested trajectories developed a similar low-rank collapse phenotype. Neither
experiment alone proves a unique or global attractor.

## 8. Frozen artifact names

- `f1_k200_freeze.json`
- `f1_k200_calibration.json`
- `f1_k200_checks.json`
- `f1_k200_vetoes.json`
- `f1_k200_stochastic_veto.json`
- `f1_k200_confirmatory.json`
- `EncoderIndependentF1K200ConfirmationResults.md`
