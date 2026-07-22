# Persistent Quantile Transport: frozen confirmatory result

Protocol: `PQT-confirmatory-v1`

Freeze commit: `7b49640dcb3e7688c60d2f0277ffbf849f2bfc21`

Raw artifact:
[`pqt_confirmatory_runs/20260721-205007-standard`](pqt_confirmatory_runs/20260721-205007-standard)

Verdict: **PASS**

## Result

The predeclared 128-knot Persistent Quantile Transport candidate passed all 13
conjunctive gates on a fresh registry created after the architecture,
hyperparameters, metrics, and thresholds were frozen.

The registry contains 20 new one-dimensional targets, three initialization
regimes, 20 paired seeds, and nine arms. All 10,800 expected trials completed.

| comparison | ED2 ratio | target-bootstrap 95% interval | SW1 ratio | target-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| PQT / selected paper `tau=.5` | **`.3487`** | `[.2223,.4940]` | **`.4582`** | `[.3495,.5684]` |
| PQT / QLD-v1 | **`.4036`** | `[.3064,.5240]` | **`.5482`** | `[.4715,.6281]` |
| PQT / gated LB-QCD | **`.4230`** | `[.3258,.5344]` | **`.5602`** | `[.4872,.6360]` |

The ED2 ratio against a per-target/initialization hindsight oracle over paper
temperatures `{.2,.5,1,2,4}` is **`.5183`**.

Equivalently, within this frozen benchmark PQT reduces target-balanced ED2 by
about 57.7% relative to the repository's previous best implementation and by
about 65.1% relative to the selected paper model.

## Breadth

PQT improves 57 of 60 target/initialization cells. Family ED2 ratios against
LB-QCD are:

| family | ratio |
|---|---:|
| connected | `.2789` |
| contaminated | `.1548` |
| equal | `.6323` |
| heavy tail | `.5694` |
| heteroscedastic | `.4388` |
| overlap | `.6068` |
| unequal | `.4158` |

Initialization ratios are:

| initialization | ratio vs LB-QCD |
|---|---:|
| concentrated | `.6076` |
| missing | `.5767` |
| far | `.2160` |

The three losing cells are retained:

| target | initialization | PQT / LB-QCD ED2 |
|---|---|---:|
| Student-t `df=2.6` | missing | `1.2134` |
| connected Gaussian shifted left | concentrated | `1.1593` |
| equal K17 mixture | missing | `1.1004` |

Their absolute cell-median ED2 values are small, but they remain genuine losses
and are not removed from any aggregate.

## Attribution

The secondary `pqt-B128-K128` arm uses exactly 153,600 target samples in every
trial and never invokes the large-batch router. It still obtains:

```text
ED2 / paper = .4112
ED2 / QLD   = .4759
ED2 / LBQCD = .4988, 95% interval [.4000,.6077]
SW1 / LBQCD = .6372, 95% interval [.5642,.7083]
```

Thus the primary improvement is not an extra-sample artifact. Persistent
transport coordinates alone account for a large gain. The resolution gate
adds a further improvement on rare separated regions.

## Budget and routing audit

For every one of the 1,200 primary candidate trials, the runner verified:

- the candidate and paired LB-QCD trial made the same routing decision;
- the candidate and paired LB-QCD trial consumed exactly the same number of
  target observations;
- routed trials used 1,024 samples for updates 1--840 and 128 thereafter;
- unrouted trials used 128 samples for all 1,200 updates.

There were 397 routed and 803 unrouted primary trials. No candidate or LB-QCD
trial diverged.

The candidate stores 128 probabilities and 128 output values. It performs no
kernel evaluations or neural forward/backward passes during training. Median
worker time was approximately `.317` seconds, versus `1.752` for LB-QCD and
`3.273` for paper `tau=.5` on this machine. These wall times are implementation-
specific; the exact sample and operation ledgers are the portable evidence.

## Integrity audit

- registry SHA-256:
  `E4B914DE6B94BB8359CCC3C3DF86E731EBE5F22855E7A6D311202D84B46DE755`;
- protocol SHA-256:
  `86D8D567B3B43B2A482527181DF95AAA4645FC0715FB81A371937BA71E4F9034`;
- raw rows: `10,800`;
- unique target/init/arm/seed keys: `10,800`;
- targets: `20`; initializations: `3`; arms: `9`; seeds: `20`;
- bootstrap replicates: `10,000`;
- `results.json` SHA-256:
  `6098984AEA960174C812E53FBDFEFDA96292187A841ECFCDCAEB11C34F17B95A`;
- a full saved-row reanalysis reproduced that hash exactly.

The artifact manifest records every source-file hash, Python/NumPy/platform
versions, command line, wall time, git revision, and dirty-worktree status.
Source snapshots are stored beside the raw rows.

## What is now established

The project has a predeclared and independently reproducible improvement over
its previous best low-dimensional drifting implementation:

> On the frozen `PQT-confirmatory-v1` one-dimensional synthetic benchmark,
> budget-matched 128-knot Persistent Quantile Transport achieved substantially
> lower ED2 and SW1 than selected paper drifting, QLD-v1, and resolution-gated
> LB-QCD, with target-level bootstrap intervals excluding no improvement,
> favorable results in every target family and initialization regime, and zero
> divergences.

## Scope boundary

This does not establish:

- superiority on images, ImageNet, or FID;
- a result for dimensions above one;
- superiority of a conventional neural generator;
- that every individual target/initialization cell improves;
- that direct empirical quantile storage is the final transferable model.

PQT is a deliberately transport-aligned, nonparametric one-dimensional
generator. The next scientific question is whether the same mechanism survives
compression into a learned monotone spline/UMNN and then extension through a
higher-dimensional convex-potential or minibatch-transport architecture.

