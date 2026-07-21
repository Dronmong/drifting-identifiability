# OA-SQD development protocol

**Program:** `OASQD-development-v1`  
**Registry:** `oasqd_development_registry.json`  
**Registry SHA-256:**
`111FA056B30931F2BEBC6C95D7DFBB4CA0C810910ECAC2A8BF693782C6E858B8`  
**Research design:**
[`OccupancyAdaptiveQuantileResearch.md`](OccupancyAdaptiveQuantileResearch.md)

## Status and scope

This is a fresh **development** campaign. It may select mechanisms and
hyperparameters, but it cannot support a confirmatory performance claim. The
two LB-QCD registries and their result artifacts remain sealed and must not be
used for OA-SQD selection.

The registry contains 16 new one-dimensional targets: separated tail and
interior rare regions, equal and heteroscedastic mixtures, legitimate remote
contamination, overlap, connected Gaussian/skew controls, a Student-t control,
and two resolution-boundary cases. Primary initializations are `missing` and
`concentrated`.

## Staged questions

### O1 -- atlas validation

Can the tail-aware persistent atlas recover compact separated regions down to
the declared resolution range while rejecting connected Gaussian, skewed, and
Student-t controls?

Atlas thresholds are chosen only from boundary/mass diagnostics and explicit
synthetic controls--not downstream generator ED2.

### O2 -- state-aware stopping

Holding full `M=1024` RSR fixed, does an occupancy controller preserve its
early coverage benefit while improving endpoint ED2 and reducing global
updates relative to the fixed 70% LB-QCD phase?

### O3 -- stratified backward estimator

On a fixed global rank table, is the importance-corrected estimator consistent
with the exact full gradient, and does training retain the O2 result with fewer
backward generator evaluations?

### O4 -- adaptive virtual resolution

Does a randomized systematic target table plus the smallest virtual batch
meeting the active-region resolution rule preserve effectiveness while
reducing total generator evaluations?

### O5 -- development tournament

Compare the selected OA-SQD candidate against:

- paper Algorithm 2 at `tau=.5`;
- a per-cell hindsight paper-bandwidth oracle, diagnostic only;
- QLD-v1;
- fixed resolution-gated LB-QCD;
- the best state-aware full-RSR arm.

## Profiles

| profile | updates | batch | seeds | endpoint sample | purpose |
|---|---:|---:|---:|---:|---|
| smoke | 40 | 32 | 1 | 512 | invariants and wiring |
| screen | 400 | 128 | 3 | 2,048 | mechanism screening |
| standard | 1,200 | 128 | 8 | 4,096 | development selection |

The atlas uses 8,192 target samples by default outside smoke. Controller probe
size, check period, pulse length, target count, and virtual-batch grid are
recorded in every manifest and result row.

## Development gates

The O5 candidate is eligible for a new frozen confirmation only if all hold:

| gate | threshold |
|---|---:|
| candidate / QLD-v1 ED2 | at most `.95` |
| candidate / selected-paper ED2 | at most `.78` |
| each primary initialization / QLD-v1 | at most `.98` |
| worst predefined family / QLD-v1 | at most `1.05` |
| coverage-event time / fixed LB-QCD | at most `1.00` |
| generator evaluations / paper | at most `3.0` |
| divergence | no worse than principal baselines |

A failed conjunction is reported as a failure. Development thresholds may not
be weakened after seeing O5 outcomes. A successful development candidate still
requires a third untouched registry and frozen protocol.

## Independence and accounting

- target-atlas, controller-probe, training-target, latent, backward-selection,
  endpoint, and metric RNG streams are separate;
- the atlas sees samples only, not target mixture metadata;
- true component metadata is used only for evaluation diagnostics already
  present in the earlier benchmark;
- atlas samples, target-table accesses, controller probes, global forwards,
  backward examples, sort work, kernel pairs, and wall time are all reported;
- the final 30% is paper-only for every hybrid arm;
- manifests snapshot every source and record the pre-run commit/status.

## Forbidden interpretations

This campaign cannot establish:

- ImageNet or real-feature superiority;
- multidimensional improvement;
- convergence of the switched neural optimizer;
- equality of atlas occupancy and equality of probability measures;
- novelty of the entire method from a repository experiment alone.
