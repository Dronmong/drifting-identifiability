# Resolution-gated LB-QCD: frozen confirmatory results

**Protocol:** `LBQCD-confirmatory-v1`  
**Registry SHA-256:**
`C2A5F01048F732EC0574A70BBD249079E4E04250EAB6D522C3449D80DC310231`  
**Protocol freeze commit:** `bbce008`  
**Raw artifact:**
[`lbqcd_confirmatory_runs/20260721-000900-test-standard`](lbqcd_confirmatory_runs/20260721-000900-test-standard)  
**Frozen verdict:** **FAIL**, solely on the minimum-effect threshold

## Result in one sentence

On 16 untouched one-dimensional targets, two primary initializations, and 20
paired seeds, resolution-gated LB-QCD produced broadly and statistically lower
ED2/SW1 than the paper model--including its hindsight bandwidth oracle--but its
`17.82%` ED2 reduction narrowly missed the protocol's required `20%` reduction.

## Frozen gate

| requirement | threshold | result | status |
|---|---:|---:|---|
| target-balanced ED2 ratio vs selected paper | at most `.80` | **`.8218`** | FAIL |
| hierarchical target-bootstrap upper endpoint | below `1` | `.8915` | PASS |
| ED2 ratio vs per-cell paper oracle | at most `.95` | **`.9437`** | PASS |
| target/initialization cell wins | at least 60% | **93.75% (30/32)** | PASS |
| worst predefined family ratio | at most `1.10` | `.8944` | PASS |
| divergence | no worse than paper | `0` vs `0` | PASS |

The correct protocol verdict is **FAIL**. Five of six checks passed, but gates
are conjunctions and `.8218 > .80`.

## Distributional outcomes

| metric | candidate / selected paper | target-bootstrap 95% interval |
|---|---:|---:|
| ED2 | **.8218** | `[.7649,.8915]` |
| SW1 | **.8679** | `[.8073,.9289]` |

Both intervals exclude `1`. The inference remains scoped to this frozen target
registry and its two initialization regimes.

The candidate also reached `.9437` against a deliberately advantaged oracle
that selected the best paper bandwidth separately in every
target/initialization cell after observing outcomes. This passes the stronger
oracle check and makes the result more than a failure of global bandwidth
tuning.

## Breadth

All predefined families were favorable:

| family | ED2 ratio vs selected paper |
|---|---:|
| connected Gaussian | .8944 |
| contaminated | .8593 |
| equal mixtures | .8642 |
| heavy tail | .7040 |
| heteroscedastic mixtures | .8287 |
| overlap | .6672 |
| unequal mixtures | .8295 |

By initialization:

- missing-mode starts: `.7752`;
- concentrated starts: `.8711`.

Thus the favorable aggregate is neither one-family nor one-initialization
selection. The two losing cells remain in the raw table and are not removed.

## What the new router added

QLD-v1 itself obtained:

```text
QLD-v1 / selected paper ED2 = .8359
QLD-v1 / paper oracle ED2   = .9599
```

Resolution gating improved the overall QLD-v1 ratio by about 1.7%, with its
clearest incremental effects on contamination (`.8854` relative to QLD-v1)
and the routed rare/heterogeneous regimes. It was neutral by construction on
unrouted connected, overlap, heavy-tail, and resolved targets.

This attribution matters. The confirmatory paper improvement is mostly the
already established quantile-to-Laplace hybrid; the target-adaptive
large-batch mechanism makes it more robust to under-resolved target regions
but is not responsible for the entire 17.8% gain.

## Routing audit

The target-only diagnostic consistently routed the intentionally
under-resolved unequal targets and rejected the resolved two-component,
K6/K12 equal, overlap, heavy-tail, and connected controls. It also routed the
K24 equal, K18 heteroscedastic, and contaminated targets, where ordinary
batches genuinely under-resolve separated regions.

Two targets showed stochastic boundary behavior:

- the min-weight `.012` K13 target routed in 60% of cells/seeds;
- the contaminated target routed in 95%.

This is not hidden. Split-half agreement prevents most false positives, but a
hard expected-count cutoff naturally produces variable decisions near the
threshold.

## Compute

Relative to selected paper:

- summed worker wall-time ratio: `1.1226`;
- generator-example-evaluation ratio: `6.5945`;
- kernel-pair ratio: `.3000`;
- divergences: `0`.

The candidate is an effectiveness result, not a sample-compute result. Batched
matrix operations kept measured wall overhead to about 12%, but the virtual
passes evaluated many more generator examples. Conversely, the quantile phase
avoided 70% of the paper's quadratic kernel pairs.

## Honest conclusion

The frozen experiment supports the following statement:

> On a new 16-target one-dimensional missing/concentrated benchmark,
> resolution-gated LB-QCD achieved about 17.8% lower target-balanced ED2 and
> 13.2% lower SW1 than the validation-selected paper model, with target-level
> bootstrap intervals excluding no improvement, favorable results in every
> target family, and a 5.6% ED2 advantage over a per-cell hindsight paper
> bandwidth oracle.

It does **not** support these stronger statements:

- that the predeclared 20% minimum effect was achieved;
- that the candidate is better from far initialization;
- that it is sample-compute efficient;
- that it works in dimensions above one or on real encoder features;
- that it improves ImageNet-scale paper results.

Because the frozen N4 gate failed, the protocol does not authorize N5
multidimensional promotion. The result should be retained as a broad scoped
improvement and a successful mechanism characterization, not retuned on this
registry.

