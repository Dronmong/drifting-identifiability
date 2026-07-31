# Quantile Safety Guard: Q2 Development Result

**Protocol:** `QGD-development-Q2-safety-v1`  
**Run:** `qgd_runs/20260721-182056-Q2-safety-screen`  
**Profile:** 8 targets x 2 initializations x 3 seeds x 4 paired arms  
**Decision:** advancement gate failed

## What was tested

Q2 repaired the principal defects found in the QGD-v1 audit.  The candidate
has one local Adam state, uses two independent current quantile gradients, and
contains no quantile momentum.  It intervenes only when both batches diagnose
quantile ascent, their gradients agree, and generator-to-target rank mismatch
exceeds a target-to-target sampling-noise floor.  Its consensus-gradient
projection requests non-ascent, and a candidate is applied only when the
correction is at most a fixed fraction of the local proposal.

The QGD registry had already been opened and investigated during Q1.  Q2 is
therefore a post-audit mechanism experiment, not sealed confirmation.

## Main result

The most permissive registered candidate was the best safety arm, but it was
effectively tied with the LB-QCD/paper-suffix baseline:

| Metric | `qsafe-permissive / lbqcd` | Gate |
|---|---:|---:|
| selected ED2 | `0.999776` | at most `0.98` |
| selected SW1 | `0.999986` | at most `0.99` |
| selected W2 | `0.999990` | descriptive |
| endpoint ED2 | `0.999562` | descriptive |
| endpoint SW1 | `0.999740` | descriptive |
| endpoint W2 | `0.999900` | descriptive |

The nominal selected-ED2 change is about `0.022%`, not a practically meaningful
improvement.  A target-level bootstrap gave an approximate 95% interval
`[0.99934, 1.00000]`; no bootstrap draw reached the predeclared `0.98` gate.

Across the 48 paired runs, permissive safety had 6 strictly better selected
ED2 outcomes, 3 strictly worse outcomes, and 39 exact ties.  Its best run-level
ratio was `0.99649` and its worst was `1.00048`.

## Why the method was nearly identical to baseline

The mechanism correctly discovered that robust harmful paper steps were rare:

- robust two-batch ascent was detected on `202 / 5760 = 3.51%` of suffix
  updates;
- only `24 / 5760 = 0.42%` of updates received a correction;
- `178 / 5760 = 3.09%` were rejected by the 25% correction cap;
- there were no divergences.

Thus only about 12% of robust-ascent detections admitted an exact consensus
non-ascent correction small enough to qualify as a safety intervention.  The
accepted corrections were too rare to change most independently selected
checkpoints.  This is consistent with the Q1 audit: after the LB-QCD prefix,
the ordinary paper suffix usually continues improving rather than erasing
global recovery.

The only visible family-level selected-ED2 movement was small:

- equal high mode count: `0.99824`;
- overlap: `0.99996`;
- heteroscedastic: `1.00001`;
- every other family rounded to `1.00000` under the cell-median aggregation.

## Compute and safety

Relative to the baseline's complete prefix-plus-suffix ledger, the safety arm
increased median generator forward calls from `9080` to `9200`, target samples
from `294400` to `302080`, and backward-equivalent examples from `294400` to
`309760`.  Most total work is the shared large-batch prefix; within the suffix,
the safety diagnostic adds an independent generator/target batch and two rank
surrogate gradient evaluations.

The implementation passed 18 QGD/Q2 invariants, including exact projection
tests and bitwise reproduction of the historical paper update whenever the
guard is inactive.  Q2 contains no second optimizer state and no uncalibrated
safe fallback.

## Decision

Q2 successfully removed QGD-v1's instability, scale mismatch, and stale
quantile-momentum failure.  It did not establish an improvement: the resulting
algorithm was a higher-cost near-copy of the baseline.

This is useful negative evidence.  It shows that merely preventing rare local
quantile-ascent steps is not the missing performance mechanism on the current
handoff.  Further work should not loosen the cap post hoc on this opened
registry.  A new candidate needs either an earlier intervention point, where
global transport remains unresolved, or a different source of benefit such as
variance reduction/adaptive bandwidth in the local field.  Any such candidate
should be developed on a separate registry.

## Artifacts

- `QuantileSafetyGuardProtocol.md`: frozen Q2 mechanism and gate.
- `quantile_guarded_drifting.py`: safety projection, step, and invariants.
- `run_quantile_guarded_development.py`: paired Q1/Q2 runner.
- `qgd_runs/20260721-182056-Q2-safety-screen/summary.json`: aggregate result.
- `qgd_runs/20260721-182056-Q2-safety-screen/rows.csv`: paired run-level data.
- `qgd_runs/20260721-182056-Q2-safety-screen/manifest.json`: source hashes,
  command, environment, and pre-run working-tree state.

