# Frozen NCJ low-dimensional improvement protocol

Status: **pre-registered and frozen before validation**, 2026-07-19.

Parent design: `IdentifiabilityDrivenImprovementPlan.md`.
Field implementation: `identifiability_drift.py`.
Runner: `run_identifiability_improvement.py`.

This protocol governs the normalized, cross-fitted, jittered (NCJ) particle
study. No threshold, target, arm, or selection rule below may be changed after
validation begins. A changed policy requires new registries and new master
seeds.

## 1. Frozen registries

The target definitions are stored as executable-data JSON rather than hidden
inside the runner:

```text
validation: identifiability_validation_registry.json
SHA256:     1DAAB55F47D4C557EB10A3960171CAA39AD2FD5BB482D11339B423CE8FBB5B97

test:       identifiability_test_registry.json
SHA256:     938B51C36178313D145DEA05A3E6FED04A033C4977B49D48CB92903D10D4A2B5
```

The validation registry contains 12 configurations. The untouched test
registry contains 16 different configurations. Neither registry repeats a
historical D3 or D3b configuration verbatim. The test registry must not be
loaded by the validation command except to verify and record its file hash.

## 2. Common particle experiment

The exact paper-style baseline is fixed at

```text
tau  = 0.35
eta  = 0.0525
mask = on
gain = P * Q
```

All arms use the exact dimension-general row/column bi-softmax affinity from
`identifiability_drift.py`. The particle cloud has `N=48`, the target batch has
`B=64`, and every run receives 400 outer updates under the standard profile.
The negative cross-fit batch has size `N`, so kernel-pair cost per field call
is identical to the paper reuse pattern.

The empirical cross-fit reference is sampled conditionally from the current
particle law by an independent with-replacement index draw. It has no
index-dependent diagonal operation. This is the particle analogue of an
independent latent reference batch; learned-generator transfer, if authorized,
must use a genuinely independent latent batch.

Constant-gain arms update using the centroid difference directly. They use a
per-query vector-norm clip of `2.0`. The clip multiplies a nonzero vector by a
strictly positive scalar and therefore does not change its zero set. Paper-gain
arms are not clipped.

## 3. Profiles

```text
smoke:
  steps=60, N=24, B=32, validation_seeds=1, test_seeds=2,
  final_reference=256, threshold_reference=128

standard:
  steps=400, N=48, B=64, validation_seeds=8, test_seeds=20,
  final_reference=1024, threshold_reference=256
```

Smoke runs test software only and cannot select or support a scientific claim.
Only a standard validation policy may be frozen for the standard test.

## 4. Initialization regimes

Every target is evaluated under four regimes:

```text
covered:
  target sample plus isotropic noise of 0.04 * target scale

missing:
  mixture targets: sample from the first declared component only;
  other targets: central Gaussian cloud of std 0.18 * target scale

far:
  central cloud translated by 3.25 * target scale along the normalized
  all-ones direction

concentrated:
  cloud centered at the target mean with std 0.015 * target scale
```

Target means/components are used only to construct declared synthetic stress
tests and to evaluate oracle diagnostics. They never enter the NCJ field or an
online policy decision.

## 5. Validation arms and selection

The constant-gain candidate grid is

```text
eta in {0.025, 0.0525}
sigma/tau in {0, 0.10, 0.25, 0.50}
norm_clip = 2.0 (fixed, not selected)
```

Validation evaluates:

1. exact paper baseline;
2. normalized-only for both candidate `eta` values;
3. cross-fit-only with paper gain and baseline `eta`;
4. jitter-only with paper gain for each positive `sigma/tau`;
5. normalized + cross-fit for both `eta` values;
6. NCJ combined for every `eta x sigma/tau` pair, including `sigma=0`.

The selected NCJ policy is the combined candidate minimizing

```text
geometric mean over target/initialization cells
  of the cell-median final ED2.
```

Target cells receive equal weight regardless of number of seeds or target
family. Ties within `1e-12` are broken by, in order:

1. smaller jitter;
2. smaller `eta`;
3. lexical arm label.

The winning `eta`, `sigma/tau`, fixed clip, registry hashes, and validation
aggregate are written to `ncj_policy_frozen.json` and committed before the
test command may run.

## 6. Frozen test arms

The untouched test evaluates exactly these arms:

| Label | Gain | Negative reference | Mask | Jitter |
|---|---|---|---|---|
| `paper` | `P*Q` | reused query cloud | on | zero |
| `normalized-only` | constant | reused query cloud | on | zero |
| `crossfit-only` | `P*Q` | independent resample | off | zero |
| `jitter-only` | `P*Q` | reused query cloud | on | frozen sigma |
| `normalized-crossfit` | constant | independent resample | off | zero |
| `ncj` | constant | independent resample | off | frozen sigma |

The two constant-gain ablations use the frozen NCJ `eta`; paper-gain ablations
use baseline `eta`. Every arm has the same number and shape of kernel matrices.
Particle experiments have no generator forward pass; this fact is recorded.

## 7. Randomness and pairing

The registry master seed determines independent streams for:

* initial particles;
* target minibatches;
* cross-fit reference indices;
* query/positive/negative jitter;
* final and threshold references;
* sliced-Wasserstein projections.

For a fixed `(target, initialization, seed)`, common streams are identical
across arms. An arm that does not use a stream must not consume it. Jitter
streams are keyed by outer step so a jitter-only and NCJ run use identical
noise at the same step.

## 8. Metrics and captured diagnostics

Each per-run row contains:

* final ED2 (primary);
* sliced Wasserstein-1;
* mixture mode coverage and mass error where available;
* nonparametric support coverage;
* on-support field residual;
* threshold event time and censoring;
* divergence and degenerate-row count;
* setup-inclusive kernel-pair count and wall time;
* generator forward count (zero in the particle study);
* particle spread and spread relative to the target;
* final medians and lower quantiles of `P*Q`, `||Delta||`, ESS+, and ESS-;
* final mean self-affinity leverage when it exists;
* realized `tau`, `eta`, `sigma`, gain, mask, cross-fit flag, and clip.

Trajectories capture the same mechanism diagnostics at step 1 and every ten
steps (or 40 evenly spaced checkpoints for nonstandard step counts). Final
particles and trajectories are stored in separate compressed NPZ files.

The threshold event is first passage of

```text
ED2 <= 0.05 * target.scale.
```

Runs not reaching it are right-censored at the final step.

## 9. Primary E4 statistic and gate

For each target/initialization cell, form paired log ratios

```text
log(ED2_NCJ / ED2_paper).
```

The point estimate is the geometric mean of cell-median ratios. The 95%
interval is a hierarchical bootstrap that resamples target/initialization
cells, then paired seeds within each selected cell. A fixed-suite row bootstrap
is diagnostic only.

NCJ passes the general particle-level gate only if all eight criteria hold:

1. target-balanced geometric-mean ED2 ratio `<= 0.80`;
2. hierarchical 95% interval upper endpoint `< 1`;
3. at least 60% of cell-median ratios are `< 1`;
4. no target family has aggregate median ratio `> 1.10`;
5. Gaussian-mixture subgroup hierarchical interval upper endpoint `< 1`;
6. non-Gaussian subgroup hierarchical interval upper endpoint `< 1`;
7. missing-mode Kaplan--Meier median recovery is no worse than paper;
8. divergence rate is no more than 2 percentage points above paper and every
   arm has the declared equal kernel-pair budget.

If criterion 5 or 6 fails, a conditional result may be reported but the
general gate fails. No criterion may be reinterpreted after the test.

## 10. Conditional learned-generator transfer

The runner must refuse the learned-generator stage unless the standard E4
gate file records `PASS=true`. If E4 fails, generator transfer is not run and
the refusal is the correct completion of this stage.

If E4 passes, the generator protocol must be frozen in a separate committed
document before execution. It must use independent latent query/reference
batches, paired initialization and optimizer streams, and exact paper
stop-gradient semantics. The particle test registry is not a generator tuning
set.

## 11. Artifacts and provenance

Every scientific run directory contains:

```text
manifest.json
rows.csv
summary.json
trajectories.npz
final_particles.npz
source_hashes.json
source_snapshots/
stdout.log
```

The manifest records the complete command/configuration, Git commit and full
dirty status, source and protocol hashes, registry hashes, platform/package
versions, seed derivation, expected/realized rows, censoring rules, wall time,
and total work. Scientific runs must start from a clean tree.

## 12. Stop and claim rules

* A failed validation candidate may be diagnosed but not promoted.
* Test targets are never recycled into selection.
* A failed E4 gate blocks learned-generator transfer under this protocol.
* A passing E4 gate supports only a held-out low-dimensional particle claim.
* A general low-dimensional algorithmic claim requires both E4 and the
  separately frozen generator gate.
* Nothing here supports an ImageNet or real-feature superiority claim.
