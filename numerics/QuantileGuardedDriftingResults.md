# Quantile-Guarded Drifting: Q1 Development Result

**Protocol:** `QGD-development-Q1-v1`  
**Run:** `qgd_runs/20260721-172034-Q1-screen`  
**Status:** completed development screen; advancement gate failed

## What was tested

The campaign tested whether optimizer-space projection could retain the local
paper field while guaranteeing a first-order fraction of the progress proposed
by exact one-dimensional rank transport. Each of 48
target/initialization/seed groups used one shared 70% LB-QCD prefix. Seven
suffix arms then received cloned handoffs and identical training, checkpoint,
and endpoint sample streams.

The registered screen covered eight new development targets, two
initializations, and three paired seeds. It produced 336 arm-level rows. Q0
passed 14 QGD-specific invariants plus the historical LB-QCD, paper-field, and
conservative-field suites. No run diverged.

## Primary result

Ratios below are relative to the same-run LB-QCD paper suffix. Lower is
better. Aggregation is the geometric mean of paired cell-median ratios across
the 16 target/initialization cells.

| Arm | Selected ED2 | Selected SW1 | Selected W2 | Endpoint ED2 | Endpoint SW1 | Endpoint W2 |
|---|---:|---:|---:|---:|---:|---:|
| fixed 10% quantile mix | 1.0232 | 1.0056 | 1.0014 | 1.0068 | 0.9974 | 0.9973 |
| periodic 4 paper : 1 quantile | 0.9850 | 0.9880 | 0.9851 | 1.2388 | 1.1304 | 1.0512 |
| paper QGD, rho=.05 | 1.0241 | 1.0080 | 0.9925 | 1.0055 | 0.9808 | 0.9910 |
| paper QGD, rho=.10 | 1.0157 | 1.0032 | 0.9891 | 1.0195 | 0.9887 | 0.9863 |
| paper QGD, rho=.20 | **0.9982** | **0.9925** | **0.9844** | 1.0434 | 1.0231 | 0.9866 |
| sharp QGD, rho=.10 | 0.9872 | **0.9562** | 1.0099 | 1.0524 | 1.0967 | 1.0800 |

The best registered paper-QGD setting was `rho=.20`, but its selected ED2 was
essentially neutral rather than the required 5% improvement. Its independent
checkpoint selector retained modest SW1 and W2 gains, while the endpoint lost
ED2 and SW1. The sharp arm had the best selected ED2/SW1 combination, but
worsened selected W2 and all endpoint metrics.

## Registered advancement decision

The `rho=.20` paper-QGD arm passed only the initialization, divergence, and
trust-radius checks. It failed:

- selected ED2: `0.9982`, required at most `0.95`;
- selected SW1: `0.9925`, required at most `0.98`;
- worst-family robustness: overlap `1.1710`, required every family at most
  `1.05`;
- safe-quantile fallback: `25.59%`, required below `5%`.

Therefore no QGD candidate advances to Q2 selection replication or Q3 sealed
confirmation. This is the predeclared stopping rule, not an implementation
failure.

## What the mechanism diagnostics say

The projection was not redundant. Its active fraction rose monotonically with
the requested quantile protection:

| Arm | Projection active | Quantile-only count | Both-active count | Safe-Q fallback |
|---|---:|---:|---:|---:|
| paper QGD, rho=.05 | 51.39% | 2,544 | 164 | 24.79% |
| paper QGD, rho=.10 | 64.05% | 3,355 | 147 | 24.95% |
| paper QGD, rho=.20 | 81.23% | 4,447 | 141 | 25.59% |
| sharp QGD, rho=.10 | 47.26% | 2,246 | 199 | 24.06% |

The zero incompatibility and zero trust-fallback rates show that the
two-constraint quadratic program was numerically well behaved. The high safe
fallback rate has a different meaning: after accumulating its own Adam
history, the raw quantile proposal was not a descent direction for the current
frozen rank surrogate about one quarter of the time. The implementation then
used the declared preconditioned safe direction. This is direct evidence that
long-lived objective-specific Adam momentum is a poor fit for a surrogate that
changes with every batch and rank matching.

The selected-versus-endpoint gap is equally important. Periodic injection and
sharp QGD can create useful intermediate checkpoints, but continued suffix
training erases or reverses those gains. Independent checkpoint selection is
doing real work; these arms are not stable endpoint successors.

## Family and initialization behavior

Paper QGD at `rho=.20` improved selected ED2 on connected-skew (`0.789`), rare
separated unequal (`0.909`), contaminated (`0.951`), and unequal (`0.971`)
families. It regressed on overlap (`1.171`), connected heavy tails (`1.136`),
equal high mode count (`1.060`), and heteroscedastic targets (`1.056`). Its
initialization aggregates were nearly neutral: `0.9997` for missing-mode and
`0.9966` for concentrated starts.

Sharp QGD improved missing-mode starts (`0.9541`) but regressed concentrated
starts (`1.0214`). This supports a narrow recovery interpretation, not a
general successor claim.

## Honest conclusion

The plan has been implemented through its registered causal test. QGD is a
soundly audited first-order projection mechanism, but this version does not
beat LB-QCD generally and should not proceed unchanged. The result isolates
two useful lessons for a future design:

1. persistent rank pressure can improve transient checkpoints, but applying it
   indefinitely destabilizes endpoints;
2. independent Adam momentum for the changing rank surrogate is the dominant
   mechanism warning, not QP infeasibility or trust-radius failure.

A future experiment should be a new, explicitly registered research plan, not
an unplanned rescue of QGD-v1. The most defensible candidate would reset or
short-horizon the quantile optimizer state and anneal/stop quantile protection
using a label-free, training-only stability signal. That hypothesis must be
tested on new development targets because the current registry has now been
opened.

## Artifacts

- `qgd_runs/20260721-172034-Q1-screen/RESULTS.md`: generated concise report;
- `summary.json`: all ratios, family/initialization splits, mechanisms, and
  median work ledgers;
- `rows.csv`: all 336 per-run outputs including exact empirical W2, coverage
  timing/censoring, worst-mode error, handoff changes, and costs;
- `checkpoint_selections.json`: Bank A/B checkpoint traces;
- `manifest.json` and `source_snapshots/`: provenance, hashes, command, git
  state, environment, and exact executed sources.

