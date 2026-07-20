# Phase C design rules: validated results (2026-07-19)

*Current results document for `DynamicsRoadmap.md` Phase C.  The historical
first pass is retained in `numerics/bench_runs/` and its procedure in
`PhaseC_DesignRules.md`.  The audit-corrected procedure is
`PhaseCValidation.md`; code is `driftbench_v2.py`; raw standard-profile runs
are in `bench_runs_v2/`; figures are in `bench_figs_v2/`.*

These are measured synthetic results, not Lean-certified theorems and not a
benchmark of the paper's trained neural model.

## Executive result

The validation pass confirms two mechanisms but rejects the first pass's
universal prescriptions:

1. **Coarse bandwidth removes the missing-mode transport barrier.**  A
   coarse-to-fine schedule is excellent in the tested 2-D targets, but it is
   not universally better than remaining coarse: fixed coarse bandwidth wins
   or ties in the tested 1-D and 5-D cases.
2. **The full generator gives an exact local Euler ceiling, not a universal
   operating step.**  The useful fraction of that ceiling varies by geometry
   and bandwidth regime.  `eta = 0.5 tau` remains a conservative heuristic,
   not a generator-derived law.
3. **Eye masking is a genuine small-particle hazard under both estimators.**
   Its effect becomes small and configuration-dependent near `N/K = 8`; no
   tested endpoint satisfied the strengthened stable-wrong-equilibrium test.

Thus Phase C now supplies defensible, qualified rules rather than the original
“anneal always / `0.5 tau` always / mask above 8 always” summary.

## Validation improvements

The new pass fixes the first pass's central defects:

* bandwidth-only and joint bandwidth/step comparisons are separate;
* multi-bandwidth arms have equal kernel-pair budgets;
* the literal normalized paper temperature set `{0.02, 0.05, 0.2}` is used
  only as a temperature baseline, not called the full paper model;
* `L` and `sigma` are estimated from unlabeled target samples as well as
  supplied by the oracle;
* the oracle bandwidth grid uses separate tuning seeds and its compute cost is
  recorded;
* threshold times are checked every step and kept right-censored;
* C2 differentiates the entire coupled `N*d` particle field;
* C3 includes both SNIS and the exact row/column bi-softmax estimator;
* equal/unequal weights, target batch size, long continuation, residual,
  spectrum, and perturbation response are recorded;
* every seed and trajectory is retained with source/configuration provenance.

## C1 — bandwidth and step size must be separated

Targets have minimum separation `L = 1`, mode width `sigma = 0.15`, eight
particles per mode, target minibatch 64, 400 base updates, and eight paired
seeds.  Initialization is a central Gaussian of width `0.25 L`, chosen to
stress missing-mode acquisition.  Entries below are median final **squared**
energy distance.

### Bandwidth-only comparison

Every arm uses `eta = 0.01 L`; multi-bandwidth arms receive fewer updates so
all use essentially the same number of kernel-pair evaluations.

| Target | fine | coarse | fine+coarse average | coarse-to-fine | held-out oracle fixed tau |
|---|---:|---:|---:|---:|---:|
| K4 d1 | 0.2555 | 0.0498 | 0.2397 | 0.1668 | **0.0414** |
| K4 d2 | 0.0339 | 0.0521 | 0.0659 | 0.0157 | **0.0106** |
| K8 d2 | 0.0307 | 0.0533 | 0.1030 | 0.0186 | **0.0086** |
| K4 d5 | 0.0379 | 0.0365 | 0.0485 | 0.0316 | **0.0176** |

This isolates the bandwidth effect.  Scheduling helps in d = 2 and modestly
in d = 5, but a tuned fixed bandwidth remains best.  Equal-weight averaging is
not competitive at equal compute in these cells.

### Joint practical policies

Here fixed bandwidth uses `eta = 0.1 tau`, while the scheduled arm uses
`eta_t = 0.1 tau_t`.  This measures the combined procedure and must not be
read as a bandwidth-only causal comparison.

| Target | fine joint | coarse joint | coarse-to-fine joint |
|---|---:|---:|---:|
| K4 d1 | 0.2743 | **0.0058** | **0.0058** |
| K4 d2 | 0.0428 | 0.0069 | **0.0047** |
| K8 d2 | 0.0425 | 0.0052 | **0.0041** |
| K4 d5 | 0.0408 | **0.0088** | 0.0194 |

The mechanism is robust: remaining permanently fine is bad when modes are
initially missing.  The best realization is geometry-dependent.  Scheduling
wins in both 2-D targets, ties in 1-D, and loses to fixed coarse bandwidth in
the tested 5-D simplex.

The unsupervised k-means estimates of `L,sigma` produced virtually identical
schedule outcomes to oracle values in all four targets.  This removes the
strongest “tuning-free only with oracle geometry” objection for these clean
mixtures, while leaving real-data estimation open.

The held-out oracle search cost 33 million kernel-pair evaluations for K4 and
88 million for K8, compared with approximately 1.23 million and 3.28 million
per evaluated single-scale run.  Avoiding a sweep can therefore be valuable
even when the schedule is not the absolute endpoint winner.

### Validated C1 rule

> Do not begin with a bandwidth far below inter-mode separation when modes are
> missing.  Start at a separation-scale bandwidth.  Anneal only when local
> refinement is needed; a fixed coarse bandwidth can be better in some
> geometries.  Estimate the relevant scales from data rather than hard-coding
> a universal temperature grid.

The next algorithmic refinement should trigger annealing from a measured
mode-mass/coverage or scale-stability diagnostic rather than a fixed 70% time
schedule.

## C2 — the corrected full-generator rule

At an exact empirical equilibrium `q = p`, C2 constructs the entire central
finite-difference Jacobian of the flattened particle field.  The truth
residual is numerically zero.  The spectral formula satisfies
`rho(I + eta* J) = 1` to displayed precision, so the implementation now
computes the intended local Euler boundary.

| Dimension | Regime | eta*/tau | Best tested operating arm | Median final ED² |
|---|---|---:|---|---:|
| d1 | tau = sigma | 14.787 | eta* boundary | 0.0197 |
| d1 | tau = L | 3.502 | 0.1 eta* (= 0.350 tau) | **0.0043** |
| d2 | tau = sigma | 13.547 | fixed 0.1 tau | **0.0083** |
| d2 | tau = L | 2.388 | 0.1 eta* (= 0.239 tau) | **0.0065** |

The boundary is exact only for the local deterministic linearization.  In the
coarse regime, operating at the boundary is much worse: ED² rises to 0.0730 in
d1 and 0.1431 in d2.  A 0.1 safety factor performs best there.  In the fine
regime the boundary becomes very large because weak, nearly neutral modes make
the spectrum ill-conditioned as a global training scale.

### Validated C2 rule

> Use the full coupled generator boundary as a local safety ceiling, never as
> the default step.  In the tested coarse regime, `0.1 eta*` is effective.  If
> computing the spectrum is too expensive, a conservative `eta` of order
> `0.1--0.5 tau` is reasonable, but no universal constant was established.

An online spectral rule must use Jacobian-vector products or a small-particle
surrogate and must account for the estimation cost.

## C3 — mask effects under SNIS and paper bi-softmax

C3 pools paired mask/no-mask comparisons over equal and unequal target mode
weights, batches 16 and 64, and six seeds.  The table reports the median ratio

```text
final ED² with mask / final ED² without mask.
```

Ratios above one mean masking is worse.

| Particles/mode | SNIS ratio | Paper bi-softmax ratio |
|---:|---:|---:|
| 1 | 23.81 | 2.60 |
| 2 | 3.33 | 1.82 |
| 4 | 2.69 | 1.25 |
| 8 | 0.995 | 1.047 |
| 16 | 0.854 | 0.956 |

The small-N hazard survives the exact Algorithm-2 affinity: masking is
decisively harmful at one or two particles per mode and remains harmful on
median at four.  By eight particles per mode the median effect is approximately
neutral; at sixteen it is mildly beneficial on median, but the interquartile
ranges still cross or approach one depending on batch and target weights.

Of 480 per-seed endpoint rows, 134 triggered the 1,600-step continuation.
**Zero** passed the strengthened `stable_wrong_candidate` gate requiring a
wrong law, small residual, contracting local spectrum, and perturbation return.
The validation therefore supports a finite-horizon quality hazard, not the
first pass's assertion that a stable wrong equilibrium was reproduced.

### Validated C3 rule

> Disable the eye mask in very small particle regimes (`N/K <= 4` in these
> tests).  Around `N/K = 8` and above, treat masking as an estimator- and
> batch-dependent choice rather than a universal improvement.  Do not infer a
> stable wrong equilibrium from a dropped mode alone.

## What Phase C now establishes

Within the declared synthetic scope:

* coarse coupling is essential for acquiring missing modes;
* schedule versus fixed coarse bandwidth is geometry-dependent;
* target scales can be recovered accurately from clean unlabeled mixture
  samples;
* the full-generator Euler formula is implemented correctly and acts as a
  ceiling rather than an operating point;
* the small-N mask hazard transfers from SNIS to the paper bi-softmax
  estimator;
* no stable wrong masked equilibrium was demonstrated by the strengthened
  tests.

It does **not** establish superiority over the paper's trained model, real-data
performance, a universal learning-rate constant, or global convergence of the
particle dynamics.  Those require encoder-feature/neural experiments and the
open quantitative theory in Phase B.
