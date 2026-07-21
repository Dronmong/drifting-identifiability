# Calibrated conservative bridge results

**Status:** optimizer-calibration stage completed; registered gate failed

**Decision:** stop before scale normalization, multiscale kernels, candidate
freezing, and Registry B.

This document reports the staged execution of
[`CalibratedConservativeBridgePlan.md`](CalibratedConservativeBridgePlan.md).
The causal protocol was committed before either run. The authoritative
artifacts are:

- smoke: [`20260721-161725-B1-smoke`](bridge_runs/20260721-161725-B1-smoke/RESULTS.md);
- registered screen: [`20260721-161822-B1-screen`](bridge_runs/20260721-161822-B1-screen/RESULTS.md).

Registry A was already exposed development data. These results are therefore
mechanism evidence, not confirmation or a claim of general superiority.

## Implementation and audit

The implementation adds a separate configurable Adam update without changing
the historical paper or QLD implementations. It supports objective-specific
learning rates, an optional cap on the flattened post-Adam parameter step,
lossless optimizer snapshots, exact restoration, and an independent
five-update paper calibration clone. The sharp phase uses exact deleted
negative references and records row mass, ESS, maximum weight share, and local
neighbor counts.

Before the screen, 39 checks passed:

- 21 conservative-field invariants;
- 7 historical training/baseline regressions;
- 9 configurable-Adam and bridge invariants;
- 2 calibration isolation/work-ledger invariants.

The smoke contained all 128 expected endpoint rows with no duplicate trial
keys. The registered screen contained all 384 expected rows
(`8 targets x 2 initializations x 3 seeds x 8 arms`) and 15,696 trajectory
rows. Every primary optimizer restoration was exact. There was no divergence
or denominator-floor activation.

## Registered endpoint result

Ratios below are target-balanced geometric means of paired cell medians. Lower
is better.

| arm | ED2 / QLD-v1 | SW1 / QLD-v1 | interpretation |
|---|---:|---:|---|
| `qld-v1` | 1.0000 | 1.0000 | registered reference |
| `qld-full` | 1.1835 | 1.0869 | paper stabilizer remains important |
| `bridge-reset-full-lr` | 0.9996 | 0.9636 | SW1 gain, but no ED2 gain and initialization harm |
| `bridge-reset-quarter` | 0.9968 | 0.9856 | small SW1 gain, essentially neutral ED2 |
| `bridge-warm-quarter` | 0.9997 | 0.9906 | essentially neutral endpoint |
| `bridge-calibrated` | **0.9997** | **0.9906** | registered primary; failed gate |
| `bridge-carry-copy` | 0.9842 | 0.9997 | best bridge ED2, but failed both aggregate thresholds |

No eligible bridge simultaneously met the registered aggregate ED2 and SW1
thresholds. The primary ED2 hierarchical bootstrap had mean ratio `0.9930`
and 95% interval `[0.9658, 1.0105]`, which includes no improvement.

## Gate audit for `bridge-calibrated`

| registered gate | observed | pass |
|---|---:|:---:|
| ED2 / QLD-v1 <= 0.98 | 0.9997 | no |
| SW1 / QLD-v1 <= 0.99 | 0.9906 | no |
| worst connected-control ED2 ratio <= 1.05 | 1.0221 | yes |
| each initialization ED2 ratio <= 1.02 | worst 1.0089 | yes |
| no divergence | none | yes |
| no denominator-floor activation | none | yes |
| median sharp/calibrated-paper step ratio in `[0.5,1.5]` | 0.3549 | no |
| 90th-percentile step ratio <= 2.0 | 0.7993 | yes |
| exact paper-state restoration | all trials | yes |

The conjunction failed, so the plan explicitly prohibits opening the scale,
multiscale, candidate-freezing, or Registry-B stages.

## What the experiment did teach us

The bridge itself retained a real transient signal. Across the 48 paired
trials, the primary 20-step bridge changed ED2 by `-0.003284` on average, and
the mean change was negative for every target. The later paper suffix added
`+0.000493` on average, erasing part of that gain; for the equal-13-mode target
it erased substantially more (`+0.008474`). This reproduces the earlier
observation that a short sharp phase can help locally without establishing a
better final algorithm.

The trust cap was not the main limiter: it activated on 30 of 960 sharp
updates (`3.125%`). The calibrated and uncapped warm-quarter endpoints were
nearly identical. Instead, the warm-quarter schedule produced a median step
only `0.355` times the independently calibrated paper step. An upper cap
cannot correct that under-stepping. Conversely, copying the carried Adam state
gave the best bridge ED2 ratio but not the required SW1 gain. Thus the simple
hypothesis "reset shock alone caused the failure" is not supported strongly
enough to advance.

Finite-particle diagnostics were numerically healthy in this screen: the
minimum recorded deleted-negative ESS was about `4.60` and the median of the
per-row minimum ESS values was about `13.30`. This does not prove estimator
quality, but it makes denominator collapse an unlikely explanation for the
registered failure.

The calibrated arm used the same 400 training updates and 25,600 training
generator-example evaluations as QLD-v1, plus 5 discarded calibration updates
and 320 calibration generator/backward examples per trial. These costs are
reported separately in the ledger.

## Honest conclusion

The implementation successfully removed optimizer-state ambiguity and exposed
the finite-particle diagnostics, but the registered calibrated bridge did not
improve the paper/QLD schedule generally enough to pass. No calibrated bridge
candidate is frozen, no `bridge_candidate.json` is created, and no fresh
registry is opened. Any further attempt should begin with a new preregistered
mechanism hypothesis rather than retroactively promoting the favorable
transient or the best exposed arm.
