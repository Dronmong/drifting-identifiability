# LB-QCD N1 development results

**Development evidence only. This registry is not sealed and these results are not a confirmatory claim.**

- selected paper baseline: `paper-0.5`
- best update-matched candidate: `rsr-M512-f0.80-u`
- advancement gate: **FAIL**

## Aggregate ratios

| arm | ED2 / selected paper | ED2 / paper oracle | SW1 / selected paper |
|---|---:|---:|---:|
| qld-v1 | 1.0489 | 1.1382 | 0.9271 |
| rsr-M1024-f0.60-e | 20.9697 | 22.7548 | 6.1554 |
| rsr-M1024-f0.60-u | 0.9488 | 1.0295 | 0.9413 |
| rsr-M1024-f0.80-e | 23.7560 | 25.7784 | 6.7314 |
| rsr-M1024-f0.80-u | 0.9328 | 1.0123 | 0.8956 |
| rsr-M512-f0.60-e | 5.8582 | 6.3569 | 2.8555 |
| rsr-M512-f0.60-u | 1.0186 | 1.1053 | 0.9675 |
| rsr-M512-f0.80-e | 6.7197 | 7.2917 | 3.1647 |
| rsr-M512-f0.80-u | 0.9291 | 1.0082 | 0.8995 |

## Advancement checks

- PASS: `unequal_ed2_improves_vs_qld_v1`
- FAIL: `no_control_family_loses_more_than_5pct_vs_qld_v1`

## Best candidate families: rsr-M512-f0.80-u

| family | vs selected paper | vs QLD-v1 |
|---|---:|---:|
| control-connected | 1.2269 | 1.1006 |
| control-contaminated | 1.2068 | 1.0057 |
| control-equal | 0.7570 | 0.9809 |
| control-heavy-tail | 0.8053 | 0.8357 |
| control-heteroscedastic | 0.6627 | 1.1138 |
| control-overlap | 1.3221 | 1.4264 |
| unequal | 0.8977 | 0.7383 |

## Interpretation rule

N1 advances only when virtual batching improves the unequal family relative to QLD-v1 without making any control family more than 5% worse. N2 advances only at ED2 ratios <= .82 against the selected paper baseline and <= .95 against the per-cell paper oracle.
