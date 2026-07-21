# Conservative-finisher results

**Protocol:** `conservative-finisher-S2-v1`

**Authoritative screen:**
`conservative_runs/20260721-153656-S2-screen`

**Decision:** S2 completed; no arm is eligible for S3 promotion. Registry B
was not created or opened.

## What was implemented and verified

The implementation includes:

- corrected Laplace mean-shift, sharp-normalized Laplace, and ordinary
  Laplace kernel-gradient fields;
- crossfit and exact reused/deleted negative references;
- slow loop references for every new field;
- stable sharp log-KDE loss evaluation;
- branch-safe QLD prefix cloning;
- Adam `carry`, `reset`, and round-tripped dual-state support;
- paired common-prefix streams, independent diagnostic streams, trajectories,
  redesigned event times, and a detailed work ledger;
- immutable Registry-A, source hashes, source snapshots, and run manifests.

The S1 gate passed all 21 field invariants and all 7 optimizer/regression
invariants. These include vectorized/slow agreement, swap antisymmetry,
translation invariance, exact equal-cloud cancellation, sharp-kernel scaling,
exact diagonal deletion, the analytical sharp-kernel derivative, field/loss
agreement, finite-difference parameter-gradient agreement, QLD compatibility,
paper-field regression checks, Adam carry/reset/dual behavior, and branch
mutation isolation.

Across the S2 screen there were no divergences and no denominator-floor
activations. The negative result below is therefore a performance result, not
a numerical crash or silent-floor artifact.

## Registered screen result

The screen used eight Registry-A targets, the two primary initializations,
three paired seeds, 400 updates, batch size 64, and a 70% shared QLD prefix.
Ratios below are target-balanced geometric means of cell-median endpoint
metrics against `qld-v1`.

| arm | ED2 ratio | SW1 ratio |
|---|---:|---:|
| `paper-0.5` | 0.9825 | 1.0928 |
| `qld-full` | 1.1589 | 1.1553 |
| `qld-paper-reset` | 1.0443 | 1.0833 |
| `qld-sharp-deleted-t0.2` | 1.0146 | 1.1108 |
| `qld-kgrad-crossfit-t1` | 1.0130 | 1.0950 |
| `qld-sharp-crossfit-t0.5` | 1.1426 | 1.1597 |
| `qld-sharp-crossfit-t1` | 1.1775 | 1.1791 |
| `qld-v1` | 1.0000 | 1.0000 |

The predeclared primary contrast, `qld-sharp-crossfit-t0.5 / qld-v1`, lost on
both metrics. The two closest ED2 arms also lost materially on SW1.

Every candidate violated the connected-control guard in at least one primary
cell. For example, the `tau=1` kernel-gradient arm's cell-median ED2 ratio was
`1.9710` on shifted-lognormal/missing, and the closest aggregate arm,
`qld-sharp-deleted-t0.2`, reached `1.4902` on
Student-t/concentrated. Therefore the registered safe-candidate ranking is
empty.

## Mechanism interpretation

Several useful distinctions emerged:

1. **The fields are stable but do not win at the fixed endpoint.** No candidate
   diverged, overflowed, or used its denominator floor. The failure is not
   explained by a defective implementation safeguard.
2. **The local phase still improves the QLD handoff.** The geometric-mean
   endpoint/handoff ED2 ratio was `0.7931` for QLD-v1,
   `0.7864` for sharp-deleted `tau=0.2`, and `0.8015` for kernel-gradient
   `tau=1`. These fields are not simply pointing away from the target.
3. **The best transient is substantially better than the endpoint.** For
   sharp-deleted `tau=0.2`, the hindsight best/handoff ED2 ratio was `0.5206`
   while its endpoint/handoff ratio was `0.7864`. Kernel-gradient `tau=1`
   showed `0.5269` versus `0.8015`. This suggests late-phase erosion or
   overtraining, but it cannot rescue the registered endpoint method: the best
   time is a per-trial hindsight oracle measured on the evaluation probe.
4. **Crossfit cost did not buy robustness.** Crossfit arms averaged 33,280
   training generator-example evaluations versus 25,600 for deleted/paper/QLD
   arms, yet the sharp crossfit variants were worse than the best deleted arm.
5. **Adam reset is not the missing fix.** Resetting the unchanged paper
   finisher yielded ED2 `1.0443` and SW1 `1.0833` versus the carried QLD-v1
   baseline.
6. **The paper finisher remains useful at the longer horizon.** Full QLD lost
   to QLD-v1 on both metrics. This reverses the short smoke ordering and is why
   the smoke result was never treated as candidate selection.

The gradient diagnostic is consistent with the observed differences: mean
parameter-gradient cosine with independent QLD was approximately `0.82` for
the paper finisher, `0.51` for sharp-deleted `tau=0.2`, `0.39` for
kernel-gradient `tau=1`, and `0.44` for sharp-crossfit `tau=1`. This is
descriptive correlation, not a causal or convergence theorem.

## Roadmap decision

The S3 advancement condition is not met. In particular:

- do not freeze a candidate from this screen;
- do not generate Registry B or C;
- do not run the optional QLD/conservative alternation from S5;
- do not describe the smoke improvement as a result;
- do not tune on the S2 endpoint until it wins in hindsight.

The most informative follow-up would be a separately registered study of
validation-based phase termination or a conservative-to-paper handoff. It
would need a new independent stopping probe and a globally frozen stopping
rule; selecting each trial's best saved step is prohibited. If that route also
fails, the evidence points away from the local field as the principal
bottleneck and toward generator parameterization or a genuinely different
scalar distribution objective.

An earlier uncommitted smoke artifact was superseded before the checkpoint
because its event and metric-work bookkeeping omitted prefix events. The
corrected committed smoke is `20260721-153547-S2-smoke`; neither smoke is used
for the decision above.
