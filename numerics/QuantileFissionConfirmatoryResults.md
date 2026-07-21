# Quantile-to-Laplace Drifting: confirmatory results

**Protocol:** `QLD-confirmatory-v1`  
**Registry SHA-256:** `7FEF49789904464A3103E09A166AD1C49A28C72DCAE00C61B87D72AA0CB1B8F8`  
**Verdict:** **pre-registered gate failed**, with a modest favorable effect on
the scoped benchmark.

The protocol and sealed registry were committed before validation. Validation
selected the paper bandwidth `tau = 0.5`; QLD remained fixed at 70% exact rank
transport followed by 30% paper Laplace refinement at `tau = 0.5`. The test
then ran 16 untouched targets, two primary initializations, 20 paired seeds,
and all five paper bandwidths: 3,840 training trials in total.

Raw artifacts:

- validation: [`qld_runs/20260720-202939-validation-standard`](qld_runs/20260720-202939-validation-standard)
- test: [`qld_runs/20260720-203253-test-standard`](qld_runs/20260720-203253-test-standard)

## Primary gate

| requirement | threshold | result | status |
|---|---:|---:|---|
| target-balanced ED2 ratio | at most .80 | **.9105** | FAIL |
| hierarchical target-bootstrap upper 95% bound | below 1 | **.9873** | PASS |
| primary-cell win fraction | at least .60 | **.7188** (23/32) | PASS |
| every predefined family ratio | at most 1.10 | worst **1.0726** | PASS |
| divergence | no worse | 0 vs 0 | PASS |

The ED2 bootstrap interval was `[.8270, .9873]`. Therefore QLD showed a
target-balanced 8.95% reduction relative to the validation-selected paper
baseline, with the interval excluding no improvement, but it did not achieve
the predeclared 20% minimum effect. The correct gate verdict is **FAIL**.

Secondary sliced-W1 ratio was `.8949`, with target-bootstrap interval
`[.8383, .9535]`.

## Where it helped

| target family | ED2 ratio |
|---|---:|
| overlap | .7469 |
| contaminated | .8113 |
| heteroscedastic | .8687 |
| equal-weight mixtures | .8945 |
| connected heavy tail | .9635 |
| unequal-weight mixtures | **1.0726** |

By initialization, the ED2 ratio was `.8918` on missing-mode starts (11/16
cell wins) and `.9295` on concentrated starts (12/16 wins). Thus the result was
not produced by only one of the two primary initialization regimes.

The unequal-weight family is the clear weakness. It became substantially worse
against the hindsight oracle paper baseline, with family ratio `1.2727`.

## Oracle paper diagnostic

The deliberately advantaged diagnostic chose the best paper bandwidth on each
test target after seeing its results. Against that oracle, QLD's ED2 ratio was
`.9774` and it won 19/32 cells. No confidence claim was pre-registered for this
post-selection diagnostic.

This near-tie changes the interpretation. QLD is reliably better than one
globally selected paper bandwidth across the heterogeneous suite, but most of
that advantage can be recovered by target-specific paper bandwidth tuning. It
is evidence for scale-robustness and cheaper global allocation, not universal
field superiority.

## Computation and dynamics

Relative to selected paper `tau=.5`:

- geometric wall-time ratio: `.498` (summed worker time 968 s vs 1,940 s);
- kernel-pair ratio: `.300`, exactly reflecting the 30% Laplace phase;
- mass-L1 ratio: `.938`, a modest calibration improvement;
- time-to-90%-weighted-reach ratio: `1.357`, so QLD was slower on this coverage
  diagnostic;
- censored coverage events: 47 for QLD vs 38 for paper;
- divergent runs: 0 for both.

The method therefore improves final ED2/SW1 and computational cost on average,
but it does not improve every notion of mode recovery. Its quantile phase
allocates final mass effectively while taking longer to satisfy the chosen
three-sigma coverage event.

## Honest conclusion

This experiment does **not** establish the strong paper-beating claim set in
the protocol. It does establish a smaller result worth retaining:

> On the sealed one-dimensional missing/concentrated fission suite, QLD
> produced approximately 9% lower target-balanced ED2 and 10.5% lower SW1 than
> a validation-selected exact paper baseline, while using 70% fewer kernel
> pairs and about half the measured training time.

That statement must remain scoped to this benchmark. It is not an ImageNet,
high-dimensional, arbitrary-initialization, or universally tuned-baseline
claim.

## Recommended decision

Do not tune QLD further on these test targets. Preserve this run as the final
assessment of `QLD-confirmatory-v1`.

If the direction is continued, use a new development split and attack the
specific unequal-weight weakness. A weighted or unbalanced transport phase is
the most mechanistically justified candidate, followed by a completely new
sealed registry. Higher-dimensional work should remain deferred because the
naive random-slice extension already failed its mechanism probe.
