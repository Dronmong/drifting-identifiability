# Persistent Quantile Transport: frozen confirmatory protocol

**Protocol ID:** `PQT-confirmatory-v1`  
**Registry:** `pqt_confirmatory_registry.json`  
**Registry SHA-256:**
`E4B914DE6B94BB8359CCC3C3DF86E731EBE5F22855E7A6D311202D84B46DE755`  
**Status at freeze:** no trial outcomes generated

## Question

Does the fixed 128-knot Persistent Quantile Transport generator improve on the
repository's best prior one-dimensional implementation, resolution-gated
LB-QCD, on new targets and under the same target-sample budget?

This protocol tests a scoped one-dimensional learned-generator claim. It does
not test higher-dimensional features, images, FID, or the paper's ImageNet
configuration.

## Frozen candidate

The primary arm is `pqt-gated-M1024-f0.70-K128`:

- latent law: scalar `Uniform(0,1)`;
- generator: monotone piecewise-linear quantile map;
- probability knots: 128 midpoint knots `(j + 1/2) / 128`;
- initialization: empirical quantiles of 4,096 untouched outputs from the
  repository's corresponding random `TanhMLP` initialization;
- initialization prior strength: one empirical-quantile batch;
- update: exact running average with one newly sampled empirical target
  quantile vector;
- ordinary target batch: 128;
- independent routing diagnostic: 4,096 target samples, using the already
  frozen split-half LB-QCD resolution rule;
- routed schedule: target batch 1,024 for updates 1--840 and 128 for updates
  841--1,200;
- unrouted schedule: target batch 128 for every update;
- no Laplace/paper suffix, checkpoint selection, learning-rate tuning, or
  outcome-dependent routing;
- fresh independent uniform latents and target samples for evaluation.

On a routed trial, the candidate and LB-QCD each consume exactly

```text
840 * 1024 + 360 * 128 = 906,240
```

training target observations. On an unrouted trial, both consume 153,600.

The secondary attribution arm `pqt-B128-K128` uses 128 target observations for
all 1,200 updates. It is reported but cannot substitute for the primary arm in
the gate.

## Baselines

Every target/initialization/seed cell runs:

1. paper Algorithm-2 field at `tau in {.2,.5,1,2,4}`;
2. validation-selected paper baseline `tau=.5`;
3. per-cell hindsight paper oracle: the smallest cell-median ED2 among the
   five paper bandwidths;
4. QLD-v1: 70% exact rank phase plus 30% paper `tau=.5` suffix;
5. gated LB-QCD: the frozen `M=1024`, 70% resolution-gated implementation;
6. the primary and secondary PQT arms above.

The active incumbent is gated LB-QCD. Beating only paper or QLD is not a pass.

## Fresh registry

The registry was created after all PQT hyperparameters and gates above were
chosen. It contains 20 new targets, none copied from QLD or LB-QCD registries:

- five unequal/separated mixtures, including `.004` and `.006` rare masses;
- three equal mixtures;
- three heteroscedastic/multiscale mixtures;
- two overlapping mixtures;
- two contaminated mixtures;
- three Student-t laws;
- two translated connected Gaussians.

Primary initializations are:

- `missing`;
- `concentrated`;
- `far`.

Each target/initialization/arm combination uses 20 paired seeds. Seeds and all
random streams are fixed by registry master seed `20260917`.

## Training and evaluation profile

```text
updates                    1,200
ordinary batch             128
evaluation samples         4,096
ED2 subsample              1,024
bootstrap replicates       10,000
primary target cells       20 * 3 = 60
paired seeds per cell      20
```

Metrics are endpoint ED2, SW1, empirical quantile W2 RMSE for PQT, weighted
mode reach, mass L1 error, event time, divergence, target samples, generator
evaluations, kernel pairs, sorting work, stored scalars, and wall time.

## Aggregation and uncertainty

1. Take the median over 20 seeds separately in every
   target/initialization/arm cell.
2. Form candidate/baseline ratios within cells.
3. Aggregate ratios with a geometric mean, giving every target and
   initialization equal weight.
4. For uncertainty, resample targets with replacement and resample paired seed
   indices independently inside every occurrence of every
   target/initialization cell.
5. Report percentile 95% intervals from 10,000 replicates.

No target, seed, family, initialization, or divergent trial may be removed.

## Conjunctive primary gate

The primary candidate passes only if every item is true:

1. target-balanced ED2 ratio versus gated LB-QCD is at most `.80`;
2. the ED2 target-bootstrap upper endpoint versus LB-QCD is below `1`;
3. SW1 ratio versus LB-QCD is at most `.85` and its bootstrap upper endpoint
   is below `1`;
4. ED2 ratio versus selected paper `tau=.5` is at most `.70`;
5. ED2 ratio versus the per-cell hindsight paper oracle is at most `.85`;
6. at least 70% of the 60 target/initialization cells have lower ED2 than
   LB-QCD;
7. every predefined family ED2 ratio versus LB-QCD is at most `1.10`;
8. each initialization-specific ED2 ratio versus LB-QCD is below `1`, and the
   far-start ratio is at most `.80`;
9. candidate divergences are no greater than LB-QCD divergences;
10. the runner verifies routed target-sample equality with LB-QCD.

These thresholds were frozen after development ratios `.6231` and `.6427`
were observed on older registries. They deliberately demand a material but
smaller effect on the new distribution of targets.

## Interpretation

- **Pass:** supports a general improvement within this explicit
  one-dimensional synthetic benchmark and architecture class.
- **Fail:** no threshold may be relaxed and no target may be removed. The
  result becomes mechanism evidence only.
- Even a pass does not imply high-dimensional or image-generation superiority.
  A parametric monotone-spline and then higher-dimensional transport program
  would still be required for that broader claim.

## Run integrity

The confirmatory runner must:

- hard-check the registry and protocol SHA-256 hashes;
- run deterministic invariants before scheduling trials;
- require an explicit `--confirm` flag for the standard profile;
- snapshot the runner, candidate implementation, baseline implementation,
  registry, and protocol into the artifact directory;
- write all raw rows before computing the summary;
- record git revision/status, Python/NumPy/platform information, CLI arguments,
  file hashes, task count, and wall time;
- refuse reanalysis with a different registry or protocol hash.

