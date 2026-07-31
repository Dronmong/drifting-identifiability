# Quantile-Guarded Drifting Smoke Results

> Development evidence only. This is not a sealed confirmation and does not support an ImageNet or paper-FID claim.

The run produced 112 arm-level rows in 3.9 seconds. All ratios below use the same-run LB-QCD arm as 1.0 and aggregate target/initialization cells by the geometric mean of paired cell-median ratios.

## Main outcomes

| Arm | Selected ED2 | Selected SW1 | Selected W2 | Endpoint ED2 | Endpoint SW1 | Endpoint W2 |
|---|---:|---:|---:|---:|---:|---:|
| fixed-mix-q0.10 | 0.9500 | 0.9800 | 0.9927 | 0.9794 | 0.9900 | 0.9891 |
| lbqcd | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| periodic-4p1q | 0.7703 | 0.9346 | 0.9404 | 0.8249 | 0.9227 | 0.9202 |
| qgd-paper-r0.05 | 0.9208 | 1.0011 | 0.9914 | 0.8837 | 0.9607 | 0.9590 |
| qgd-paper-r0.10 | 0.9252 | 1.0012 | 0.9909 | 0.8707 | 0.9564 | 0.9539 |
| qgd-paper-r0.20 | 0.8742 | 1.0007 | 0.9831 | 0.8477 | 0.9426 | 0.9407 |
| qgd-sharp-r0.10 | 0.8764 | 0.9637 | 0.9654 | 0.8635 | 0.9041 | 0.9386 |

Lower ratios are better. Checkpoint selection used independent Bank A ranking followed by Bank B earliest-within-one-SE confirmation.

## Mechanism diagnostics

| Arm | Projection active | Safe-Q fallback | Trust fallback | Incompatible | Divergences |
|---|---:|---:|---:|---:|---:|
| fixed-mix-q0.10 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| lbqcd | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| periodic-4p1q | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| qgd-paper-r0.05 | 0.299 | 0.118 | 0.000 | 0.000 | 0 |
| qgd-paper-r0.10 | 0.312 | 0.118 | 0.000 | 0.000 | 0 |
| qgd-paper-r0.20 | 0.375 | 0.122 | 0.000 | 0.000 | 0 |
| qgd-sharp-r0.10 | 0.361 | 0.125 | 0.000 | 0.000 | 0 |

## Registered advancement decision

Best registered paper-QGD arm: `qgd-paper-r0.20`.

- PASS — `selected_ed2_vs_lbqcd_at_most_0.95`
- FAIL — `selected_sw1_vs_lbqcd_at_most_0.98`
- FAIL — `all_family_ed2_ratios_at_most_1.05`
- PASS — `all_init_ed2_ratios_at_most_1.02`
- PASS — `no_divergence`
- FAIL — `safe_quantile_fraction_below_0.05`
- PASS — `trust_fraction_below_0.10`

Overall advancement gate: **FAIL**.

The complete family splits, initialization splits, active-set counts, outcome diagnostics, and compute ledger are in `summary.json`; per-run values are in `rows.csv`; checkpoint selection traces are in `checkpoint_selections.json`.
