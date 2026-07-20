# Fresh low-dimensional mask attribution: results

*Corrective follow-up to the audited D0--D3 study. Protocol frozen at commit
`45f4f43`; validation policy frozen at commit `1783728`. Exact executed-source
snapshots, full manifests, rows, and trajectories are under
`numerics/lowdim_runs/20260719-200021-D3b-factor-validation/` and
`numerics/lowdim_runs/20260719-200822-D3b-fresh-test/`.*

## Verdict

The pre-declared fresh held-out gate **FAILED**. The selected conditional mask
rule does not provide a 20% improvement on the primary median paired-error
aggregate:

```text
aggregate paired ED² ratio:             1.000
target-aware hierarchical 95% CI:       [0.646, 1.000]
cells with paired-median ratio > 1.10:   0 / 14
missing-start KM time, base / modified:  51 / 53
non-Gaussian paired median ratio:        0.905
non-Gaussian hierarchical 95% CI:        [0.558, 1.000]
```

The strict superiority criteria (`ratio <= 0.8`, CI high endpoint `<1`, and
missing-start time no worse) do not hold. The full learned-generator D4 run
therefore remains blocked.

The experiment nevertheless closes the main attribution question from the
audit: **the historical ring/moons improvements came from the mask change, not
from geometry-matched bandwidth or step size.**

## Validation factorial

The fresh validation study crossed bandwidth, mask, and step under both
missing and covered starts on six diverse target laws (576 trials total).

| Arm | Geometric mean of cell-median ED² |
|---|---:|
| fixed tau, **auto mask**, fixed eta | **0.003943** |
| fixed tau, auto mask, scaled eta | **0.003943** |
| geometry tau, auto mask, fixed eta | 0.004119 |
| geometry tau, auto mask, scaled eta | 0.004341 |
| fixed tau, mask on, fixed eta (base) | 0.004876 |
| fixed tau, mask on, scaled eta | 0.004876 |
| geometry tau, mask on, fixed eta | 0.005121 |
| geometry tau, mask on, scaled eta | 0.005263 |

The winner improved the validation aggregate by 19.1% (`0.809` ratio) and
changed only one factor:

```text
tau  = 0.35                 (same as tuned base)
eta  = 0.0525               (same as tuned base)
mask = on iff N >= 8 K_hat  (only modification)
```

Thus the earlier `sqrt(sigma_hat L_hat)` bandwidth hypothesis is not supported
as a performance improvement by this diverse validation suite. Geometry
bandwidth is worse both with the mask held on and with the auto mask.

## Fresh held-out attribution

The confirmation used seven new target configurations, two initializations,
and 20 paired seeds (1,400 labeled arm trials; 280 unique base-versus-mask
pairs). The `tau-only` and `step-only` labels are exactly bitwise-identical to
base because the frozen winner retained those base factors; `mask-only` and
`combined` are likewise identical. This is an internal reproducibility check.

The paired cell-median ED² ratios for the mask policy are:

| Fresh target | Missing | Covered |
|---|---:|---:|
| 1-d K6 Gaussian | 1.000 | 1.000 |
| 2-d K4 unequal Gaussian | 1.000 | 1.000 |
| Banana | 1.000 | 1.000 |
| Circles | **0.564** | **0.510** |
| Moons | **0.676** | **0.486** |
| Ring | **0.497** | **0.523** |
| Sine ridge | 1.000 | 1.000 |

Most cells are exact ties because the estimated rule leaves the mask on. The
primary median over all paired rows is therefore exactly one. The
pre-registered statistic properly reports no global superiority.

As an explicitly exploratory descriptive statistic, the cell-balanced
geometric mean of the 14 cell-median ratios is `0.767`, with a hierarchical
bootstrap interval approximately `[0.652,0.908]`. This exposes the sparse,
conditional benefit that the median gate deliberately does not call an
aggregate win. It must not replace the failed pre-declared gate post hoc.

## What happens when the rule switches the mask off

| Target | Mask-off trials | Median ED² ratio among mask-off trials | Trials worse than base |
|---|---:|---:|---:|
| 1-d K6 Gaussian | 8/40 | 1.090 | 8/8 |
| Circles | 38/40 | 0.529 | 0/38 |
| Moons | 34/40 | 0.620 | 0/34 |
| Ring | 26/40 | 0.477 | 0/26 |
| Sine ridge | 2/40 | 0.613 | 0/2 |

The rule is highly effective on curved multi-branch geometry, where
silhouette k-means usually returns `K_hat > 6`, and harmful when it
occasionally overestimates the true six components of the 1-d Gaussian target.
This explains both the strong conditional gains and the absence of a universal
win.

The mechanism is now causally isolated: bandwidth and step are fixed, paired
randomness is identical, and the only changed computation is removal of the
eye mask in selected runs.

## Secondary diagnostics

Across the 14 cells, exploratory cell-balanced geometric-mean ratios are:

```text
ED²:                  0.767
sliced W1:            0.940
on-support residual:  1.049
wall time:             1.009
```

Distributional error improves on the curved cells, but the final drift
residual is modestly larger and missing-start threshold time is two steps
slower in aggregate. The mask policy should therefore be viewed as a
finite-horizon distribution-quality tradeoff, not as evidence of faster
relaxation to the same equilibrium.

There were no divergences. Every training trial used exactly `2,150,400`
kernel pairs; setup and evaluation wall time are recorded per run. All source
and snapshot hashes match their manifests.

## Audit repairs completed

The corrective study resolves the implementation shortcomings recorded in the
audit:

* fresh curved and connected validation targets prevent the fixed baseline
  from being selected on Gaussian mixtures alone;
* bandwidth, mask, and step are crossed factorially;
* validation and confirmation targets are disjoint from historical D3 and
  from each other;
* target-aware hierarchical inference replaces row-wise pseudo-replication;
* setup costs are included in per-run accounting;
* realized geometry estimates and mask decisions are retained;
* ED², sliced W1, coverage, mass error, residual, censoring, divergence, wall
  time, cost, final particles, and trajectories are retained;
* each manifest starts clean and snapshots/hashes every executed source plus
  the frozen protocol;
* the historical “paper temperature,” “zero tuning,” “within ±25%,” and
  geometry-attribution overclaims are corrected in the status documents.

## Research conclusion and next move

There is still no defensible aggregate claim that the modified low-dimensional
procedure outperforms the tuned base implementation. The full D4 generator
benchmark should not be run for this failed global policy.

There is, however, a reproducible conditional design finding:

> With the tuned bandwidth and step held fixed, disabling the eye mask can
> reduce finite-horizon distributional error by roughly 32–52% on curved
> ring/circle/moon targets, while being exactly inert when the mask remains on.
> A cluster-count-only trigger can misfire on finite Gaussian mixtures.

Any further empirical development must use another fresh split. The natural
candidate is a mask trigger based on affinity leverage or self-mass rather than
`K_hat` alone, because those observables distinguish harmful self-suppression
directly and avoid treating an overclustered Gaussian mixture as curved
geometry. Until such a trigger is selected on new data and passes a new held-
out gate, B1 may proceed only as an independent metastability-theory project,
not as an explanation of a demonstrated aggregate performance win.
