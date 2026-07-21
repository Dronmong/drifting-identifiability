# Large-Batch Quantile-Calibrated Drifting: development results

**Program:** `LBQCD-development-v1`  
**Status:** successful development result; not a frozen confirmatory claim  
**Primary artifact:**
[`lbqcd_runs/20260720-234857-n1-standard`](lbqcd_runs/20260720-234857-n1-standard)

## Executive result

The useful successor to QLD-v1 is not unconditional large-batch transport and
not adaptive bandwidth selection. It is **resolution-gated large-batch
quantile drifting**:

1. inspect an independent target-only quantile sample;
2. detect compact separated regions that an ordinary batch is expected to
   represent fewer than eight times;
3. use virtual-batch `M=1024` Run-Sort-ReRun during the 70% quantile phase only
   when that under-resolution certificate fires;
4. otherwise use QLD-v1 exactly;
5. finish with the paper Laplace field at the already validated `tau=.5`.

On the full development profile--12 new targets, missing and concentrated
initializations, eight paired seeds, and 1,200 optimizer updates--this method
obtained:

| comparison | ED2 ratio | interpretation |
|---|---:|---|
| gated LB-QCD / selected paper (`tau=.5`) | **.7948** | 20.5% lower target-balanced ED2 |
| gated LB-QCD / per-cell hindsight paper oracle | **.8685** | favorable even after target-specific paper tuning |
| QLD-v1 / selected paper | .8424 | new mechanism improves on QLD-v1, not only the paper arm |
| gated LB-QCD / QLD-v1 | .9435 | approximately 5.7% further reduction |

The target-bootstrap 95% interval for the selected-paper ED2 ratio was
`[.7063,.8910]`; the SW1 ratio was `.8456`, with interval `[.7679,.9229]`.
The candidate won 19/24 target/initialization cells and had zero divergences.

This is the strongest low-dimensional development result in the repository.
It is still not a claim that the paper has been beaten generally: the registry
was used for mechanism development, dimensions above one are untested, and a
small far-start diagnostic remains strongly unfavorable.

## 1. Implementation and correctness

The implementation lives in:

- [`lbqcd.py`](lbqcd.py): rank transport, exact RSR gradient accumulation,
  resolution diagnosis, experimental noise restoration, and bandwidth
  selectors;
- [`run_lbqcd_development.py`](run_lbqcd_development.py): registry execution,
  paired baselines, work accounting, bootstrapping, immutable artifacts, and
  reanalysis;
- [`lbqcd_development_registry.json`](lbqcd_development_registry.json): the
  mutable development registry, disjoint from `QLD-confirmatory-v1`.

The RSR implementation performs two passes through each virtual batch but one
Adam update. Microbatch gradients are divided by the full virtual batch size,
summed, and only then applied. Fast invariants verify:

- `M=batch` RSR exactly reproduces an ordinary QLD rank update;
- changing the backward microbatch partition preserves the update up to
  floating-point summation tolerance;
- rerunning before the update reproduces the stored outputs exactly;
- the target router rejects connected quantiles and accepts a compact rare
  separated component;
- no hidden optimizer update is made per microbatch.

## 2. Development registry

The registry contains six unequal mixtures with nominal minimum weights from
`.005` to `.05`, plus equal, heteroscedastic, overlapping, contaminated,
Student-t, and connected-Gaussian controls. The primary initializations are
`missing` and `concentrated`.

No target identifier or parameterization from the sealed QLD-v1 validation or
test registry was reused. This makes it suitable for development after the
QLD-v1 result, but it is not sealed evidence because its results informed the
final routing rule.

## 3. What unconditional large batches established

At the 400-update screen, pure `M=1024` RSR changed the overall ED2 ratio from
QLD-v1's `1.0489` to `.9446` against the selected paper baseline. More
importantly, its unequal-family ratio relative to QLD-v1 was `.7628`.

This validates the original diagnosis: small batches were materially
under-resolving rare quantile intervals.

Pure RSR nevertheless failed the N1 guardrail. Relative to QLD-v1 it worsened
the overlapping control by `1.4775` and the heavy-tail control by `1.2293`.
Large batches suppressed useful stochastic symmetry-breaking and applied an
expensive global correction even where the local/noisy path was already good.

Generator-evaluation-matched pure RSR was also rejected. Reducing optimizer
updates enough to pay for the two virtual-batch passes produced ED2 ratios
between roughly `6.3` and `22.7`. Large batches do not substitute for optimizer
updates in this generator.

## 4. Repairs tested before the router

### Phase fractions

Warm fractions `.60`, `.70`, and `.80` were tested at `M=512` and `M=1024`.
None preserved all controls. The problem was not a single bad handoff time.

### Periodic global corrections

RSR pulses every 4, 8, or 16 warm updates retained more QLD stochasticity but
still damaged the overlap control by factors from approximately `1.46` to
`1.90` relative to QLD-v1.

### Mean-zero noise restoration

The conditionally unbiased gradient experiment used

```text
(1-lambda) * full_virtual_gradient + lambda * random_global_match_subgradient.
```

Values `lambda in {.25,.50,.75}` improved some controls but did not repair
overlap. This showed that the global map itself, not only reduced gradient
variance, was unnecessary in that regime.

These rejected variants remain in the code and artifacts as an audit trail;
none is promoted.

## 5. Resolution-gated RSR

The final N1 router uses 4,096 independent target samples, split into two
disjoint diagnostic halves. In each half it:

1. sorts the samples;
2. trims extreme order-statistic gaps;
3. requires a gap at least 20 times the median interior spacing and at least
   1.5% of the central target span;
4. requires the gap to dominate compact neighborhoods on both sides, rejecting
   isolated heavy-tail spacings;
5. estimates the masses of the separated quantile regions;
6. fires only if the smallest region has fewer than eight expected samples in
   the ordinary batch;
7. requires both half-sample diagnoses to fire.

This rule uses no component labels, target family, mixture weights, or test
metrics. It routed all six unequal targets and the contaminated target,
usually routed the heteroscedastic target, and consistently rejected the
equal, overlap, Student-t, and connected controls in the standard run.

### Standard family results

| family | gated / selected paper | gated / QLD-v1 |
|---|---:|---:|
| connected | .7246 | 1.0000 |
| contaminated | .8877 | .8793 |
| equal | .8267 | 1.0000 |
| heavy tail | .6824 | 1.0000 |
| heteroscedastic | .6251 | .8447 |
| overlap | .8358 | 1.0000 |
| unequal | .8335 | .9353 |

Thus every predefined family beat the selected paper arm, and no control lost
more than 5% relative to QLD-v1. On non-routed targets, equality with QLD-v1 is
by construction rather than an approximate empirical coincidence.

## 6. Honest compute ledger

Relative to the selected paper arm in the standard run:

- summed worker wall-time ratio: `1.2217`;
- generator-example-evaluation ratio: `7.8906`;
- kernel-pair ratio: `.3000`;
- divergences: `0`.

The large virtual batches are efficient matrix operations on this CPU, so an
almost eight-fold generator-evaluation count translated into only 22% more
measured wall time. Nevertheless, the candidate is an effectiveness win, not
a sample-compute win. Its 70% rank phase still removes 70% of the paper's
quadratic kernel work.

Both work currencies must remain reported. Wall time alone would hide the
extra generator evaluations; kernel pairs alone would make the method appear
artificially cheap.

## 7. Bandwidth calibration result (N2)

The held-out cosine-alignment selector was decisively wrong for separated
targets. It frequently chose `tau=1` or `2` when hindsight paper training
preferred `.2` or `.5`; gated ED2 worsened from `.9059` to `.9779` at the
screen profile.

An optimizer-aware selector then cloned the complete generator and Adam state,
made one candidate update at each bandwidth, and selected the smallest
independent held-out empirical W2. This improved the ratio to `.9350`, but
still lost to fixed `.5` and failed both N2 gates. Adaptive bandwidth is
therefore rejected for the current candidate.

N3 rare-quantile weighting was not activated: the gated N1 method already
repaired the unequal family without sacrificing contaminated or heavy-tail
controls. Adding weights would confound a successful mechanism.

## 8. Far-start boundary

A small diagnostic on an unequal mixture, the overlap control, and the
Student-t control gave:

```text
gated LB-QCD / paper-.5 ED2 = 3.3089
QLD-v1 / paper-.5 ED2       = 3.5331
```

The router modestly improves QLD-v1 but does not repair its far-translation
failure. Missing and concentrated starts are the legitimate scope of the
candidate; arbitrary initialization is not.

## 9. Decision

The frozen candidate for a new confirmatory experiment should be:

```text
diagnostic target sample = 4096, split-half agreement
ordinary batch           = 128
virtual batch            = 1024 when routed
under-resolution cutoff  = fewer than 8 expected samples
quantile fraction        = 0.70
refinement               = paper Algorithm 2, tau = 0.5
optimizer/architecture   = unchanged TanhMLP + Adam
```

The next legitimate claim-bearing step is a new immutable registry and at
least 20 paired seeds. The development registry and its thresholds must not be
retuned again or reused as confirmatory evidence.
