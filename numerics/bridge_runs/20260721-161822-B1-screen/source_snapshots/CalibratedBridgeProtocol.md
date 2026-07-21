# Calibrated conservative bridge protocol

**Protocol ID:** `calibrated-bridge-B1-v1`

**Status:** frozen before the B1 smoke and Registry-A screen

**Governing plan:**
[`CalibratedConservativeBridgePlan.md`](CalibratedConservativeBridgePlan.md)

This protocol tests whether the previous conservative-finisher failure was
caused by an abrupt Adam reset and an excessively long sharp-only suffix. It
does not test scale normalization, multiscale kernels, memory banks, mixed
divergences, or an adaptive phase duration.

## Registered schedule

For the screen profile:

```text
total updates:  400
QLD prefix:     280
sharp bridge:    20
paper suffix:   100
batch:            64
paired seeds:      3
```

The smoke profile uses 120 updates, batch 32, and one paired seed while
retaining a 20-update bridge. Both use all eight exposed Registry-A targets
and the `missing` and `concentrated` initializations.

Every sharp arm uses the sharp-normalized Laplace field, exact reused/deleted
negative references, and `tau=0.5` in raw coordinates.

## Registered arms

- `qld-v1`: 70% QLD, then carried paper Algorithm 2;
- `qld-full`: QLD for the full horizon;
- `bridge-reset-full-lr`: reset sharp Adam, repository LR;
- `bridge-reset-quarter`: reset sharp Adam, one-quarter LR;
- `bridge-warm-quarter`: reset sharp Adam, one-quarter LR with 20-step linear
  warm-up;
- `bridge-calibrated`: warm-quarter plus a parameter-step trust cap;
- `bridge-carry-copy`: copy the QLD Adam state into sharp, preserving an
  untouched copy for the paper suffix;
- `qld-sharp20-only`: calibrated bridge endpoint diagnostic, ineligible for
  selection because it uses fewer total updates.

Every full bridge restores the QLD/paper Adam state saved before the bridge,
while retaining the parameters learned during the bridge.

## Trust-cap calibration

At the handoff, five paper updates are executed on an independent discarded
clone and dedicated streams. If their median parameter-step norm is
`s_paper`, the calibrated cap is `1.25*s_paper`. The clone may not mutate the
training prefix. All calibration examples, forwards, kernel pairs, backward
examples, and optimizer updates are charged to each deployable arm that uses
the cap.

The sharp LR on bridge update `j` is
`0.25 * repository_lr * j/20` for warm/calibrated arms.

## Registered decision gates

`bridge-calibrated` advances only if all hold:

- ED2/QLD-v1 at most `0.98`;
- SW1/QLD-v1 at most `0.99`;
- every connected-control cell ED2/QLD-v1 at most `1.05`;
- each primary-initialization ED2/QLD-v1 at most `1.02`;
- no divergence or denominator-floor activation;
- median sharp/calibrated-paper committed-step ratio in `[0.5,1.5]`;
- 90th-percentile sharp/calibrated-paper committed-step ratio at most `2.0`;
- exact paper optimizer-state restoration.

Ratios are target-balanced geometric means of cell medians. Registry A is
already exposed, so passing permits only the next development stage. Failure
prohibits scale normalization, multiscale implementation, and Registry B under
this plan.

The step ratios divide every committed sharp update by the same trial's
five-update, carried-paper calibration median `s_paper`, then pool those
per-update ratios. This uses the predeclared independent calibration stream;
it is not estimated from sparse trajectory checkpoints or endpoint metrics.

## Interpretation

The B1 screen is a causal optimizer/schedule diagnosis. It cannot establish
confirmation, high-dimensional performance, stochastic convergence, or that
conservativity alone improves generator training.
