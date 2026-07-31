# Quantile-Guarded Drifting Screen Results

> Development evidence only. This is not a sealed confirmation and does not support an ImageNet or paper-FID claim.

The run produced 336 arm-level rows in 301.0 seconds. All ratios below use the same-run LB-QCD arm as 1.0 and aggregate target/initialization cells by the geometric mean of paired cell-median ratios.

## Main outcomes

| Arm | Selected ED2 | Selected SW1 | Selected W2 | Endpoint ED2 | Endpoint SW1 | Endpoint W2 |
|---|---:|---:|---:|---:|---:|---:|
| fixed-mix-q0.10 | 1.0232 | 1.0056 | 1.0014 | 1.0068 | 0.9974 | 0.9973 |
| lbqcd | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| periodic-4p1q | 0.9850 | 0.9880 | 0.9851 | 1.2388 | 1.1304 | 1.0512 |
| qgd-paper-r0.05 | 1.0241 | 1.0080 | 0.9925 | 1.0055 | 0.9808 | 0.9910 |
| qgd-paper-r0.10 | 1.0157 | 1.0032 | 0.9891 | 1.0195 | 0.9887 | 0.9863 |
| qgd-paper-r0.20 | 0.9982 | 0.9925 | 0.9844 | 1.0434 | 1.0231 | 0.9866 |
| qgd-sharp-r0.10 | 0.9872 | 0.9562 | 1.0099 | 1.0524 | 1.0967 | 1.0800 |

Lower ratios are better. Checkpoint selection used independent Bank A ranking followed by Bank B earliest-within-one-SE confirmation.

## Mechanism diagnostics

| Arm | Projection active | Safe-Q fallback | Trust fallback | Incompatible | Divergences |
|---|---:|---:|---:|---:|---:|
| fixed-mix-q0.10 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| lbqcd | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| periodic-4p1q | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| qgd-paper-r0.05 | 0.514 | 0.248 | 0.000 | 0.000 | 0 |
| qgd-paper-r0.10 | 0.640 | 0.249 | 0.000 | 0.000 | 0 |
| qgd-paper-r0.20 | 0.812 | 0.256 | 0.000 | 0.000 | 0 |
| qgd-sharp-r0.10 | 0.473 | 0.241 | 0.000 | 0.000 | 0 |

## Registered advancement decision

Best registered paper-QGD arm: `qgd-paper-r0.20`.

- FAIL — `selected_ed2_vs_lbqcd_at_most_0.95`
- FAIL — `selected_sw1_vs_lbqcd_at_most_0.98`
- FAIL — `all_family_ed2_ratios_at_most_1.05`
- PASS — `all_init_ed2_ratios_at_most_1.02`
- PASS — `no_divergence`
- FAIL — `safe_quantile_fraction_below_0.05`
- PASS — `trust_fraction_below_0.10`

Overall advancement gate: **FAIL**.

The complete family splits, initialization splits, active-set counts, outcome diagnostics, and compute ledger are in `summary.json`; per-run values are in `rows.csv`; checkpoint selection traces are in `checkpoint_selections.json`.
