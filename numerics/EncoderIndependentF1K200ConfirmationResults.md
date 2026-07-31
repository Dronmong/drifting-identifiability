# Corrected encoder-independent F1 K=200 confirmation

## Result

**F1 FAILS in both regimes: replay 0/3, stochastic 0/3.**

This is the fresh confirmation specified in
`EncoderIndependentF1K200ConfirmationProtocol.md`. It uses three new units,
only the primary `random_generator` arm and its paired `real_data` control, one
shared kernel calibration, globally disjoint replay/control allocations, and a
terminal checkpoint fixed at K=200 before the new outcomes were generated.

Artifact: `encoder_independent_drifting/f1_k200_confirmatory.json`, SHA-256
`7c8dfb478febac9441281794c965860eec81215e888121e059f087dcbc35d2d8`.
The confirmation itself took 417.6 seconds on the recorded RTX 4050 laptop GPU.

## Preflight

Every precondition passed under the exact frozen source manifest
`f1_k200_freeze.json` (SHA-256 `8270ff963b7382af...`):

- null calibration: `GO`, with 0/200 exceedances for both decisive null states
  and `p_null_upper = 0.014867 < 0.025` at the fixed 0.05 gate;
- one-step equivalence: relative L2 error exactly 0;
- replay reproducibility: relative L2 error exactly 0;
- historical Phase-26 K=40 regression: recall 0.71680 and KID +0.00041, both
  inside their frozen tolerances;
- all validation and smoke checks passed;
- all five replay/collapse veto statistics calibrated successfully;
- stochastic nearest-training-image veto: `GO`, threshold 0.423038;
- every JSON artifact matches its SHA-256 sidecar and records the source hashes
  in the freeze manifest.

## Gate values at K=200

The gate requires, in the same unit: primary recall above 0.05, every applicable
veto, and the paired real-data control above 0.5.

| unit | regime | real-data control | primary recall | primary rank | veto | replicate |
|---:|---|---:|---:|---:|---|---|
| 100 | replay | 0.6226 | 0.0005 | 5.69 | fail | fail |
| 100 | stochastic | 0.6348 | 0.0024 | 6.00 | fail | fail |
| 101 | replay | 0.6519 | 0.0039 | 5.81 | fail | fail |
| 101 | stochastic | 0.6587 | 0.0029 | 5.75 | fail | fail |
| 102 | replay | 0.6587 | 0.0010 | 5.93 | fail | fail |
| 102 | stochastic | 0.6592 | 0.0020 | 5.79 | fail | fail |

All six positive controls pass, so neither regime is void. All six primary
recalls are far below 0.05. Every primary also fails the independently
calibrated effective-rank veto (`rank >= 6.8925`), while the other applicable
veto comparisons pass. Thus the result does not depend on a borderline recall
or on only one failure diagnostic.

The replacement reference-side intervals all have upper endpoints below
0.0077. They are explicitly diagnostics over covered reference indicators,
not total generated-manifold uncertainty and not gate conditions.

## Trajectory reading

Median primary recall over the three new units:

| regime | K=0 | K=40 | K=100 | K=200 |
|---|---:|---:|---:|---:|
| replay | 0 | 0 | 0 | 0.0010 |
| stochastic | 0 | 0 | 0 | 0.0024 |

The real-data controls remain useful throughout the same window:

| regime | K=0 | K=40 | K=100 | K=200 |
|---|---:|---:|---:|---:|
| replay | 0.7158 | 0.7124 | 0.6875 | 0.6519 |
| stochastic | 0.7158 | 0.7134 | 0.7031 | 0.6587 |

This is the contrast F1 was designed to test: within a horizon where the update
still preserves substantial coverage from a valid real-data start, it does not
bring the deployed random-generator start into a coverage-bearing state.

The result is not explained by a numerically dead kernel. Across the primary
terminal cells, denominator-floor fraction and collapsed-row fraction are both
zero, while ESS fraction is at least 0.5896.

For scale, the matched 512-vs-2,048 real-vs-real diagnostic has recall 0.7339,
precision 0.7207, KID -0.00019 and effective rank 9.05. Absolute FID is not
compared with published values at this small sample count.

## Honest conclusion

The corrected experiment supports this narrow claim:

> Under the tested raw-pixel, smooth-Laplace, RMS-normalized, R11-corrected
> finite-sample update, the new random-generator starts do not acquire detected
> CIFAR-10 manifold coverage by K=200, in either replay or stochastic sampling.

This cleanly selects the prescribed-bridge/new-objective branch for the current
update. It does **not** prove a unique global attractor, say that no encoder could
change the dynamics, adjudicate the population converse, or reproduce the
paper's full model.

The earlier K=20,000 run remains separate exploratory evidence that the tested
starts developed a similar long-horizon low-rank collapse phenotype. Its
overstated confirmatory/global-attractor language should not be imported into
this result.

