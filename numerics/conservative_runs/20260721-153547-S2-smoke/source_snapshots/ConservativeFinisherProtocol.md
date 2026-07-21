# Conservative-finisher mechanism-screen protocol

**Protocol ID:** `conservative-finisher-S2-v1`

**Frozen before S2:** 2026-07-21

**Scope:** low-dimensional learned-generator mechanism development. This is
not an untouched confirmation and cannot support an ImageNet, real-feature,
or general superiority claim.

The governing design document is
[`QLDSharpConservativeImplementationPlan.md`](QLDSharpConservativeImplementationPlan.md).
This protocol fixes the choices required to run stages S1 and S2.

## Registered data and profiles

Registry A is `conservative_registry_a.json`, identified by SHA-256
`48AA09809A78413E2C2C2314618677FDC3B870C638EAD69C05E205DB6089AF12`.
Its component labels may be used only for evaluation.

| profile | updates | batch | paired seeds | endpoint samples | ED samples |
|---|---:|---:|---:|---:|---:|
| smoke | 60 | 32 | 1 | 512 | 256 |
| screen | 400 | 64 | 3 | 2,048 | 512 |

The primary initializations are `missing` and `concentrated`. `far` is an
optional separately reported stress diagnostic. The QLD prefix is 70% of the
update horizon. Trajectories are evaluated every 25 updates, at the handoff,
for each of the first five post-handoff updates, and at the endpoint.

## Registered fields and arms

The exact repository paper field and exact QLD rank update are reused. The new
field definitions and reference semantics are implemented in
`conservative_finishers.py` and audited by its slow reference functions.

The fixed baseline arms are:

- `paper-0.5`;
- `qld-v1` (70% QLD then paper at `tau=0.5`, Adam carried);
- `qld-full`;
- `qld-paper-reset`.

For each `tau` in `{0.2, 0.5, 1.0}`, S2 also runs:

- corrected mean shift with crossfit negatives and reset Adam;
- sharp Laplace with exact deleted negatives and reset Adam;
- sharp Laplace with crossfit negatives and reset Adam;
- Laplace kernel-gradient with crossfit negatives and reset Adam.

The predeclared research contrast is `qld-sharp-crossfit-t0.5 / qld-v1`.
Other arms and bandwidths are mechanism diagnostics used by the frozen S3
selection rule; they are not hindsight baselines.

## Streams and optimizer semantics

Each target/initialization/seed group has separate streams for initialization,
training latents, positive samples, crossfit latents, endpoint evaluation,
event metrics, and gradient diagnostics. Branches consume paired arrays from
the same streams. Crossfit negative outputs are detached. Exact deletion
removes diagonal terms rather than applying a large finite penalty.

`carry` preserves all Adam moments and its step index. `reset` zeros both
moment buffers and restores the step index to zero. The code also supports a
round-tripped dual state, but S2 does not alternate objectives.

## Fixed diagnostics

The label-free event occurs when both ED2 and SW1 are at most 25% of their
step-zero values on the independent probe. For mixture targets, the occupancy
event additionally requires every component with target mass at least 0.005
to have nonzero observed occupancy and component mass-L1 at most 0.25.

S2 records endpoint ED2, SW1, target-quantile CDF error, component diagnostics,
best and handoff ED2, five-step ED2 change, field and gradient norms, Adam
descent alignment, QLD/finisher gradient cosine, denominator minima and floor
activations, generator work, sample work, pair/sort work, and wall time.

## Mechanism elimination and S3 rule

An arm is unsafe if it diverges or activates denominator floors on an ordinary
screen batch. An arm is eliminated if it loses to QLD by more than 10% on
either connected control. No S2 result is confirmation.

Among safe, non-eliminated arms, S3 uses this lexicographic rule:

1. lowest target-balanced geometric mean of cell-median ED2 ratio versus
   QLD-v1;
2. when within one hierarchical-bootstrap standard error, lower standalone
   generator-evaluation cost;
3. if still tied, exact deletion before crossfit, then smaller bandwidth.

The selected arm must be written to immutable candidate JSON before Registry B
is created or run. The Registry-B advancement thresholds remain those in
Section 12 of the governing plan.

## Interpretation guardrail

S2 can diagnose whether a conservative local finisher and optimizer reset
improve a shared QLD prefix. It cannot establish confirmation, novelty,
population convergence of Adam, or a general performance improvement over the
paper.
