# Fresh low-dimensional attribution protocol

*Pre-registered corrective follow-up to the D0--D3 audit. This study does not
reuse any target configuration from the historical held-out suite for policy
selection or confirmation. Code: `lowdim_attribution.py`; shared exact
Algorithm-2 field: `lowdim_drift.py`; runs: `lowdim_runs/<id>/`.*

## Question

The historical frozen policy changed three quantities at once:

```text
bandwidth: fixed tau*  -> sqrt(sigma_hat L_hat)
step:      fixed eta*  -> 0.15 tau
mask:      always on   -> on iff N >= 8 K_hat
```

It had large exploratory wins on rings and moons, but the mask was frequently
off on precisely those targets. The follow-up asks which factor, if any,
causes a repeatable improvement on fresh low-dimensional distributions.

## Fixed baseline and common training configuration

The historical D0 baseline is retained without retuning:

```text
tau = 0.35
eta = 0.0525
mask = on
```

All arms use the exact dimension-general row/column bi-softmax Algorithm-2
estimator with the empirical particle cloud reused as negatives. Standard
profile: `N=48`, target batch `B=64`, 400 updates, 1024 final reference
samples, 256 threshold-reference samples. Missing and covered initializations
are both tested. Randomness is paired across arms.

## Fresh target splits

No configuration below appeared in the historical D3 suite.

### Attribution validation

Used only to select one factorial candidate:

* `AV-1d-K4-uneq`: unequal 1-d four-Gaussian mixture, `L=0.8`, `sigma=0.18`;
* `AV-2d-K5-hetero`: five-Gaussian ring, `L=1.1`, component sigmas
  `0.10,0.14,0.18,0.22,0.26`;
* `AV-ring`: radius `0.8`, width `0.07`;
* `AV-circles`: radii `0.30,0.85`, width `0.06`;
* `AV-moons`: scale `0.75`, noise `0.10`;
* `AV-banana`: connected quadratic-banana law with curvature `0.45`.

Six paired seeds per cell are used under the standard profile.

### Fresh held-out confirmation

Touched only after the factorial policy is frozen:

* `AT-1d-K6-eq`: equal 1-d six-Gaussian mixture, `L=0.9`, `sigma=0.11`;
* `AT-2d-K4-uneq`: unequal 2-d four-Gaussian mixture, `L=1.2`, `sigma=0.22`;
* `AT-ring`: radius `1.25`, width `0.035`;
* `AT-circles`: radii `0.45,1.25`, width `0.05`;
* `AT-moons`: scale `1.25`, noise `0.05`;
* `AT-banana`: connected banana law with curvature `0.70`;
* `AT-sine`: connected noisy sinusoidal ridge.

Twenty paired seeds per cell are used under the standard profile.

## Validation factorial

Cross all eight combinations:

```text
bandwidth in {fixed 0.35, geometry sqrt(sigma_hat L_hat)}
mask      in {always on, auto: on iff N >= 8 K_hat}
step      in {fixed 0.0525, scaled 0.15 tau}
```

`K_hat`, `L_hat`, and `sigma_hat` come from silhouette-selected k-means on
256 unlabeled setup samples. Mode labels and true target parameters are never
read by a policy. The validation score is the geometric mean across the 12
target/initialization cells of each cell's median final ED-squared.

The minimum-score arm is frozen. The test runner also constructs three
component arms from its factors (`tau-only`, `mask-only`, `step-only`) so the
fresh confirmation remains an attribution experiment rather than a single
combined comparison.

## Fresh held-out gate

The primary comparison is frozen combined policy versus frozen baseline.

1. Aggregate paired ED-squared ratio must be at most `0.8`.
2. The upper endpoint of a hierarchical 95% bootstrap interval must be below
   `1`. The bootstrap resamples target cells, then paired seeds within cells.
3. At most 20% of cell-median paired ratios may exceed `1.10`.
4. Missing-initialization Kaplan--Meier median threshold time must not worsen.
5. On non-Gaussian target cells, the hierarchical interval must favor the
   modification (`hi < 1`).

The historical row bootstrap is retained only as a fixed-suite diagnostic.
The target-aware interval is the inferential result.

If this gate fails, the full learned-generator D4 run remains blocked. The
held-out targets may not be recycled into another selection loop.

## Required artifacts

Every run must retain:

* commit, full git status, tracked-diff hash, command line, configuration,
  package versions, and snapshots/hashes of this protocol and all executed
  Python sources;
* per-seed final ED-squared, sliced W1, coverage, mass error, on-support
  residual, divergence, threshold time/censoring, setup-inclusive kernel-pair
  cost, and per-run wall time;
* realized `K_hat`, `L_hat`, `sigma_hat`, `tau`, `eta`, and mask choice;
* compressed trajectories containing threshold ED, mode diagnostics,
  bandwidth, step, and mask;
* validation aggregates, the frozen policy, per-arm held-out statistics, and
  the pre-declared gate result.

## Claim discipline

Passing supports only a low-dimensional empirical-particle claim. Failure is
recorded as a negative result. Named target-family wins are exploratory unless
the relevant factorial contrast and target-aware interval support their
mechanism. “No per-target sweep” must not be rewritten as “zero tuning cost.”
