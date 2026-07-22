# Persistent Quantile Transport: development and replication results

**Program:** `transport-aligned-generator-development-v1`  
**Candidate:** `gated-pqt-M1024-f0.70`, 128 monotone knots  
**Status:** development success plus cross-registry replication; not a new
frozen confirmatory claim

## Result in one sentence

Replacing the non-monotone two-dimensional-latent MLP with a persistent
one-dimensional monotone quantile map reduced target-balanced ED2 by about
`38%` relative to the repository's best LB-QCD implementation on its
development registry and by about `36%` on the separate prior confirmatory
registry, under the same 70%-virtual-batch target-sample schedule.

## Primary development result

Artifact:
[`transport_aligned_runs/20260721-203517-standard-K128-standard`](transport_aligned_runs/20260721-203517-standard-K128-standard)

The standard mutable LB-QCD registry has 12 targets, missing and concentrated
initializations, eight paired seeds, batch 128, and 1,200 updates.

| comparison | ED2 ratio | target-bootstrap 95% interval |
|---|---:|---:|
| PQT / selected paper (`tau=.5`) | **`.4952`** | `[.3922,.6383]` |
| PQT / QLD-v1 | **`.5878`** | `[.4510,.7759]` |
| PQT / gated LB-QCD | **`.6231`** | `[.4817,.8111]` |

The SW1 ratio versus LB-QCD is `.6657`, with interval `[.5935,.7799]`.
Every predefined family is favorable, both initializations improve, 19/24
target/initialization cells improve, and there are no divergences.

Family ED2 ratios versus LB-QCD:

| family | ratio |
|---|---:|
| connected | `.9863` |
| contaminated | `.3009` |
| equal mixtures | `.8556` |
| heavy tail | `.7765` |
| heteroscedastic | `.8171` |
| overlap | `.4970` |
| unequal mixtures | `.5913` |

Initialization ratios are `.6403` for concentrated and `.6063` for missing.

## Cross-registry replication

Artifact:
[`transport_aligned_runs/20260721-203819-replication-standard-cross-registry-replication`](transport_aligned_runs/20260721-203819-replication-standard-cross-registry-replication)

After the candidate and 128-knot choice were fixed, it was replayed without
tuning on the 16-target, two-initialization, 20-seed registry previously used
for LB-QCD's frozen test.

| comparison | ED2 ratio | target-bootstrap 95% interval |
|---|---:|---:|
| PQT / selected paper (`tau=.5`) | **`.5281`** | `[.4293,.6399]` |
| PQT / QLD-v1 | **`.6318`** | `[.5062,.7703]` |
| PQT / gated LB-QCD | **`.6427`** | `[.5167,.7884]` |

The SW1 ratio versus LB-QCD is `.7044`, with interval `[.6189,.7926]`.
All seven target families and both initializations improve, 29/32 cells win,
and there are no divergences.

This is useful replication but not a fresh confirmatory claim: the registry
already existed and its earlier LB-QCD outcomes were visible in the repository.
A formal promotion still requires a newly generated, untouched registry and a
frozen protocol.

## Matched-sample attribution

The ordinary `pqt-B128` arm uses exactly the paper/QLD target-sample count:
`128 * 1200 = 153,600` target observations per trial.  On the development
registry it obtains:

```text
ED2 / paper = .6513
ED2 / QLD   = .7731
ED2 / LBQCD = .8194, bootstrap interval [.7031,.9757]
SW1 / LBQCD = .8398, bootstrap interval [.7719,.9263]
```

Thus persistent transport coordinates improve the incumbent even without
LB-QCD's extra target samples.  The gated arm adds resolution for rare
intervals but is not solely responsible for the win.

## Budget correction

The first exploratory high-resolution arm accidentally used a 1024-sample
target batch for all 1,200 routed updates.  That result (`.5443` versus LB-QCD)
is retained as a sample-heavier headroom diagnostic, not the primary result.

The promoted development candidate uses 1024 samples for the first 70% of
routed updates and 128 for the final 30%, exactly matching LB-QCD:

```text
840 * 1024 + 360 * 128 = 906,240 target samples.
```

Its `.6231` development and `.6427` replication ratios are the budget-correct
figures.

## Knot-count falsifier

The result is insensitive to table size:

| knots | gated PQT / LB-QCD ED2 | matched B128 PQT / LB-QCD ED2 |
|---:|---:|---:|
| 128 | `.6231` | `.8223` |
| 256 | `.6181` | `.8199` |
| 1024 | `.6167` | `.8194` |

The gain therefore does not depend on a 1024-knot lookup table.  The 128-knot
map has 128 learned output values, substantially fewer degrees of freedom than
the repository's roughly 1,185-parameter one-dimensional `TanhMLP`.

## Far-start diagnostic

Artifact:
[`transport_aligned_runs/20260721-203631-screen-far-diagnostic`](transport_aligned_runs/20260721-203631-screen-far-diagnostic)

On the exact three-target far-start diagnostic where LB-QCD previously scored
`3.3089` relative to paper, the budget-matched 128-knot PQT scores:

```text
ED2 / paper = .3709
ED2 / QLD   = .1050
ED2 / LBQCD = .1121
SW1 / paper = .3809
```

This supports the architectural diagnosis: a persistent quantile map replaces
the initial marginal directly, whereas the local learned-generator dynamics
can remain trapped far from the target.

## What changed

QLD and LB-QCD recompute an empirical rank matching on each fresh latent
minibatch, pass the resulting displacement through a shared non-monotone MLP,
and then discard the pairing.  PQT instead stores the one-dimensional order in
the generator:

```text
u ~ Uniform(0,1)
G(u) = monotone linear interpolation of ordered knots
q <- running average(q, empirical target quantiles)
```

Convex combinations of ordered quantile vectors stay ordered.  The generator
does not have to discover a latent ordering, maintain batch-dependent labels,
or approximate a fission map through saturated tanh layers.  The experiment
therefore changes the representation of transport, not the Laplace bandwidth
or field gain.

## Compute and capacity

For the primary 128-knot gated arm on the development registry, median trial
work is:

```text
target samples             906,240  (matched to gated LB-QCD)
stored scalars             256      (128 probabilities + 128 values)
kernel pairs               0
median worker wall time    0.246 s
```

The runner also charges 4,096 evaluations of the original random MLP to match
its initial marginal and 4,096 fresh generated samples for evaluation.  PQT
does not perform neural-network forward/backward passes during training.

Wall time is implementation- and machine-specific.  The defensible claim is
the exact sample/work ledger, not a universal speedup factor.

## Why this is not yet “the paper is beaten generally”

The result is intentionally narrow:

- one-dimensional distributions only;
- a nonparametric monotone generator rather than the paper's image generator;
- synthetic mixture, connected, overlap, contamination, and heavy-tail
  families;
- development and previously known replication registries;
- ED2, SW1, empirical quantile error, reach, and mass diagnostics—not FID or
  ImageNet quality.

It does establish a stronger scoped fact than previous attempts: on two broad
low-dimensional registries, a transport-aligned generator beats both the tuned
paper field and LB-QCD by a large margin under an audited target-sample budget.

## Next decision

Do not add a paper suffix: pure PQT already wins broadly, and prior suffix
programs repeatedly erased good transient states.  The next steps are:

1. freeze the 128-knot, one-prior-batch, 70%-gated candidate;
2. create a new untouched registry with new parameter draws, tail stresses,
   connected controls, and far initialization;
3. predeclare ratios versus selected paper, per-cell paper oracle, QLD, and
   LB-QCD, with hierarchical bootstrap intervals;
4. only after confirmation, replace the running empirical map with a learned
   monotone spline/UMNN to test whether the mechanism survives a transferable
   parametric implementation;
5. treat higher dimensions as a separate convex-potential or minibatch-OT
   research program.

