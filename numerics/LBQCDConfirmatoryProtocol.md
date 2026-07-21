# Frozen protocol: resolution-gated LB-QCD confirmation

**Protocol ID:** `LBQCD-confirmatory-v1`  
**Registry SHA-256:**
`C2A5F01048F732EC0574A70BBD249079E4E04250EAB6D522C3449D80DC310231`  
**Frozen before outcomes:** yes

## Question

On new one-dimensional targets under missing or concentrated initialization,
does resolution-gated virtual-large-batch Quantile-to-Laplace Drifting reduce
final distributional error relative to both the deployable paper baseline and
a hindsight bandwidth oracle?

This protocol does not test dimensions above one, far initialization,
ImageNet-scale generation, or sample-compute superiority.

## Frozen candidate

The candidate is exactly the N1 development winner:

```text
ordinary batch                    128
target diagnostic sample          4096
diagnostic split                  two disjoint half-samples; both must fire
interior gap / median spacing     >= 20
gap / central target span         >= .015
gap / adjacent compact span       >= 1.5
under-resolution threshold        < 8 expected ordinary-batch samples
virtual batch when routed         1024
RSR backward microbatch           128
quantile phase                    first 70% of 1200 updates
refinement phase                  final 30%
refinement field                  exact paper Algorithm 2
refinement bandwidth              tau = .5
generator                         repository TanhMLP
optimizer                         existing Adam, unchanged
```

When the diagnostic does not fire, the entire candidate is definitionally
QLD-v1. When it fires, each quantile update performs a no-cache virtual forward
pass, one global monotone match, a cache-bearing rerun in microbatches, and one
Adam update after summing gradients over all 1,024 samples.

No bandwidth selector, rare-quantile weight, noise-restoration term, periodic
pulse, or target-family label is allowed.

## Frozen registry

`lbqcd_confirmatory_registry.json` contains 16 targets that are disjoint from
both `QLD-confirmatory-v1` and `LBQCD-development-v1`:

- seven unequal mixtures, including minimum weights from `.004` to `.10`;
- three equal mixtures;
- two heteroscedastic mixtures;
- one overlapping mixture;
- one legitimate remote-contamination mixture;
- one Student-t heavy-tail control;
- one connected Gaussian control.

The two primary initialization regimes are `missing` and `concentrated`.
There are 20 paired seeds per target/initialization/arm.

## Baselines

The deployable paper baseline is frozen at `tau=.5`, selected on the earlier
QLD-v1 validation registry before this registry existed. The runner also trains
paper arms at `tau in {.2,.5,1,2,4}`. Their per-cell hindsight minimum is an
explicitly advantaged diagnostic oracle, not a deployable selection rule.

QLD-v1 is retained as a mechanism baseline. All arms share the same initial
parameters, seed identifier, architecture, update count, ordinary batch size,
and final evaluation draws. Different batch-size arms necessarily consume
different training streams and the work difference is reported.

## Outcomes

Primary metric: final squared energy distance (ED2). For every
target/initialization cell, take the median over 20 seeds; aggregate cell ratios
with a target-balanced geometric mean.

Uncertainty: hierarchical target bootstrap with target resampling and paired
seed resampling within each target/initialization cell, 10,000 replicates.

Secondary outcomes:

- sliced Wasserstein-1;
- target-mass L1 and weighted reach diagnostics;
- cell and family ratios;
- routing rate;
- divergence;
- summed worker wall time;
- generator-example evaluations, unique latent samples, target samples,
  kernel pairs, and diagnostic samples.

## Confirmatory gate

All conditions must pass:

1. candidate/selected-paper ED2 ratio at most `.80`;
2. target-bootstrap ED2 upper 95% endpoint below `1`;
3. candidate/per-cell-paper-oracle ED2 ratio at most `.95`;
4. candidate wins at least 60% of target/initialization cells;
5. every predefined family ED2 ratio versus selected paper is at most `1.10`;
6. candidate divergence count is no greater than selected paper.

The `.80` threshold is inherited from the failed QLD-v1 protocol rather than
chosen from this registry. The oracle threshold comes from the N2 advancement
rule written before the development implementation.

## Decision rules

- **Pass:** report a scoped one-dimensional missing/concentrated improvement,
  with compute costs and far-start exclusion beside the headline.
- **Fail:** retain the development mechanism but make no paper-beating claim;
  do not retune on this registry.
- Regardless of outcome, do not reuse these targets for another confirmatory
  test or silently remove unfavorable families.

Stage N5 (informative multidimensional projections) is allowed only after a
pass. Its results require a separate protocol and cannot be pooled into this
gate.
