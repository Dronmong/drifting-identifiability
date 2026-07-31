# Quantile-Guarded Drifting Smoke Results

> Development evidence only. This is not a sealed confirmation and does not support an ImageNet or paper-FID claim.

The run produced 64 arm-level rows in 2.7 seconds. All ratios below use the same-run LB-QCD arm as 1.0 and aggregate target/initialization cells by the geometric mean of paired cell-median ratios.

## Main outcomes

| Arm | Selected ED2 | Selected SW1 | Selected W2 | Endpoint ED2 | Endpoint SW1 | Endpoint W2 |
|---|---:|---:|---:|---:|---:|---:|
| lbqcd | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| qsafe-balanced | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| qsafe-permissive | 0.9998 | 0.9999 | 1.0000 | 0.9985 | 0.9991 | 0.9993 |
| qsafe-strict | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Lower ratios are better. Checkpoint selection used independent Bank A ranking followed by Bank B earliest-within-one-SE confirmation.

## Mechanism diagnostics

| Arm | Projection active | Safe-Q fallback | Trust fallback | Incompatible | Divergences |
|---|---:|---:|---:|---:|---:|
| lbqcd | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| qsafe-balanced | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| qsafe-permissive | 0.007 | 0.000 | 0.000 | 0.000 | 0 |
| qsafe-strict | 0.000 | 0.000 | 0.000 | 0.000 | 0 |

## Registered advancement decision

Best registered paper-QGD arm: `None`.

- FAIL — `selected_ed2_vs_lbqcd_at_most_0.98`
- FAIL — `selected_sw1_vs_lbqcd_at_most_0.99`
- PASS — `all_family_ed2_ratios_at_most_1.05`
- PASS — `all_init_ed2_ratios_at_most_1.02`
- PASS — `no_divergence`
- PASS — `intervention_fraction_below_0.20`
- PASS — `cap_rejected_fraction_below_0.10`

Overall advancement gate: **FAIL**.

The complete family splits, initialization splits, active-set counts, outcome diagnostics, and compute ledger are in `summary.json`; per-run values are in `rows.csv`; checkpoint selection traces are in `checkpoint_selections.json`.
