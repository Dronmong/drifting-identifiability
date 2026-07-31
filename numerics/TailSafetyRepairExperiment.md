# Projected-Tail and Safety Repair Experiment

## Status

This development protocol was written before running the repaired factorial
screen. It uses the already-consumed 16-target development registry and cannot
support a fresh confirmation claim.

## Repairs under test

1. Preserve the existing per-particle local safety rule as the guard factor.
   Its exact scope is frozen-rank mean squared residual over the active
   projection bank.
2. Replace displacement-only ordering with a rank-aware priority:

   ```text
   priority =
     0.5 * percentile(displacement squared norm)
     + 0.5 * percentile(mean projected-rank extremeness).
   ```

3. Select the upper 10% priority stratum and distribute it across every
   64-sample student microbatch without replacement.
4. Record tail and bulk student residuals separately.
5. Re-evaluate the same latent cohort after all micro-updates and record direct
   teacher-to-student rare-core retention.
6. Charge the teacher-rank projection/sort, top-k, and reorder work in the
   explicit ledger.

Exact permutation means every particle remains included once. No inverse
weight is required for the registered 512/64 equal-size partition; the repair
changes update ordering, not sample frequency.

## Factorial arms

All arms use the confirmed dimension-adaptive rollout:

| Arm | Safety | Rank-aware balance |
|---|---:|---:|
| `cta-exact-adaptive-rollout` | no | no |
| `cta-exact-adaptive-rollout-safe` | yes | no |
| `cta-exact-adaptive-rollout-rank-balanced` | no | yes |
| `cta-exact-adaptive-rollout-safe-rank-balanced` | yes | yes |

The rollout horizon remains one/two/four steps in 2D/4D/8D/16D. All other
optimizer, atlas, direction, representative, local-field, and budget settings
remain unchanged.

## Evaluation

- consumed profile and concentrated initialization;
- 16 development targets spanning four dimensions and four families;
- paired ED2, held-out SW1, and training quantile RMSE against the unmodified
  adaptive rollout;
- results by dimension and family;
- rare-core teacher count, same-cohort post-student count, and final output
  core mass;
- tail/bulk residual ratio;
- measured wall time and corrected projection/sort work.

An arm is worth fresh confirmation only if it improves both aggregate ED2 and
held-out SW1 without a compensating high-dimensional rare-core regression.
The screen will not tune the `0.5` priority mixture or the 10% tail fraction.

