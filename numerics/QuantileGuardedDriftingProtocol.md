# Quantile-Guarded Drifting development protocol

**Protocol ID:** `QGD-development-Q1-v1`  
**Scope:** Q0 invariant audit and Q1 low-dimensional mechanism screen  
**Status:** completed development screen; advancement gate failed; not a
confirmatory superiority claim

The completed decision is recorded in
[`QuantileGuardedDriftingResults.md`](QuantileGuardedDriftingResults.md). No
candidate advanced to Q2 or Q3.

The governing analysis and implementation specification is
[`QuantileGuardedDriftingResearchPlan.md`](QuantileGuardedDriftingResearchPlan.md).

## Registered registry and target scope

The development registry is `qgd_development_registry.json`, with SHA-256:

```text
3CA4E4AC966513271D1FD1B5C7B8D721926C8EE7572F18A1395FE453F8C71701
```

It contains eight one-dimensional targets not duplicated in prior repository
registries. Both `missing` and `concentrated` generator initializations are
used. Target component labels are evaluation-only and never enter a field,
router, projection, or selector.

## Profiles

| profile | updates | batch | paired seeds | endpoint / ED samples | Bank A | Bank B |
|---|---:|---:|---:|---:|---:|---:|
| smoke | 60 | 32 | 1 | 512 / 256 | 2 x 256 | 2 x 512 |
| screen | 400 | 64 | 3 | 2,048 / 512 | 4 x 1,024 | 4 x 2,048 |

The shared LB-QCD prefix occupies 70% of the update horizon. The frozen
resolution diagnostic uses 1,024 target-only samples in smoke and 4,096 in
screen. When it activates, the prefix uses `M = 1024` Run-Sort-ReRun with the
ordinary training batch as microbatch; otherwise it uses ordinary direct QLD.

The local paper bandwidth is fixed at `tau = 0.5`. No bandwidth selection or
target-aware suffix routing is allowed.

## Shared-handoff contract

For each target/initialization/seed group:

1. initialize one generator;
2. train the LB-QCD prefix once;
3. clone parameters and complete Adam state into every suffix arm;
4. give every branch the same latent batches, positive batches, endpoint
   samples, selection banks, and checkpoint schedule.

Mutation of one branch must not affect another. QGD objective-specific Adam
states begin as independent exact copies of the carried handoff state.

## Registered arms

1. `lbqcd`: historical carried-state paper suffix;
2. `periodic-4p1q`: four paper updates followed by one direct QLD update;
3. `fixed-mix-q0.10`: normalized 10% quantile / 90% paper proposal mixture;
4. `qgd-paper-r0.05`;
5. `qgd-paper-r0.10`;
6. `qgd-paper-r0.20`;
7. `qgd-sharp-r0.10`: exact-deleted sharp/log-KDE secondary arm.

The QGD metric is the positive diagonal preconditioner from the quantile Adam
proposal. The trust factor is `2.0`; the local non-ascent constraint is active.
The primary rank constraint is never silently dropped.

## Checkpoint selection

Checkpoints are saved at the handoff, every five suffix updates in smoke or ten
in screen, and the endpoint. All arms receive the same selection opportunity.

Bank A ranks every eligible checkpoint with the equal-log-weight normalized
ED2/SW1 score. Bank B independently evaluates the top three Bank-A leaders.
The selected output is the earliest checkpoint within one Bank-B standard
error of the best Bank-B mean.

Eligibility requires:

- weighted coverage no more than `0.05` below the handoff;
- mass-L1 no more than `0.10` above the handoff.

Endpoint metrics use another independent paired sample stream. The report must
show both endpoint and selected results. Selection work is separate from
training work.

## Required Q0 invariants

Before any empirical run:

- all historical LB-QCD invariants pass;
- the paper and conservative field invariant suites pass;
- inactive, one-active, two-active, singular, and opposing-gradient QGD
  projection tests pass;
- simulated Adam proposals match cloned real steps numerically;
- an explicit disabled-guard update reproduces the historical carried paper
  update bitwise;
- the two-bank earliest-within-one-SE selector is deterministic;
- the registry hash and cross-registry disjointness checks pass.

## Recorded mechanism diagnostics

Every arm records endpoint and selected ED2/SW1, coverage, mass error, selected
step, projection active-set counts, gradient cosine, correction norm,
quantile/local directional derivatives, trust-cap activation, safe-quantile
fallback, incompatibility, singular candidates, divergence, kernel pairs,
sort work, forward examples, backward examples, target samples, selection
work, endpoint work, and wall time.

## Q1 advancement gate

Among `qgd-paper-r0.05`, `qgd-paper-r0.10`, and `qgd-paper-r0.20`, select the
lowest target-balanced geometric mean of cell-median selected ED2 relative to
`lbqcd`. It advances only if all checks hold:

- selected ED2/LB-QCD at most `0.95`;
- selected SW1/LB-QCD at most `0.98`;
- every family ED2/LB-QCD at most `1.05`;
- both initialization aggregates at most `1.02`;
- no divergence;
- safe-quantile fallback below `5%`;
- trust-cap activation below `10%`.

The screen is not confirmation even if every check passes. A winner must be
frozen, replicated with a new development registry, and then tested once on a
new sealed confirmatory registry as described in the governing plan.

## Interpretation boundary

Q1 can determine whether optimizer-aware persistent rank protection improves a
shared LB-QCD handoff and whether independent checkpointing retains a real
transient gain. It cannot establish ImageNet performance, high-dimensional
performance, novelty, population convergence of Adam, or a finite-step
monotonicity theorem.
