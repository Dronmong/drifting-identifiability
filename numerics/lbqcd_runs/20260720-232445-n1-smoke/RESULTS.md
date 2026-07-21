# LB-QCD N1 development results

**Development evidence only. This registry is not sealed and these results are not a confirmatory claim.**

- selected paper baseline: `paper-0.5`
- best update-matched candidate: `rsr-M1024-f0.70-u`
- advancement gate: **FAIL**

## Aggregate ratios

| arm | ED2 / selected paper | ED2 / paper oracle | SW1 / selected paper |
|---|---:|---:|---:|
| qld-v1 | 1.3997 | 1.4686 | 1.1722 |
| rsr-M1024-f0.70-e | 6.1530 | 6.4559 | 2.2816 |
| rsr-M1024-f0.70-u | 1.3483 | 1.4147 | 1.1604 |
| rsr-M512-f0.70-e | 6.1538 | 6.4567 | 2.2816 |
| rsr-M512-f0.70-u | 1.3655 | 1.4327 | 1.1568 |

## Advancement checks

- PASS: `unequal_ed2_improves_vs_qld_v1`
- FAIL: `no_control_family_loses_more_than_5pct_vs_qld_v1`

## Best candidate families: rsr-M1024-f0.70-u

| family | vs selected paper | vs QLD-v1 |
|---|---:|---:|
| control-connected | 3.2246 | 0.9432 |
| control-contaminated | 0.5813 | 0.6582 |
| control-equal | 1.4642 | 0.9664 |
| control-heavy-tail | 2.5277 | 1.7742 |
| control-heteroscedastic | 1.2459 | 0.9059 |
| control-overlap | 1.5881 | 1.2172 |
| unequal | 1.1749 | 0.9035 |

## Interpretation rule

N1 advances only when virtual batching improves the unequal family relative to QLD-v1 without making any control family more than 5% worse. N2 advances only at ED2 ratios <= .82 against the selected paper baseline and <= .95 against the per-cell paper oracle.
