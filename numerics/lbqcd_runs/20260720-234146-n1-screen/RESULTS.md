# LB-QCD N1 development results

**Development evidence only. This registry is not sealed and these results are not a confirmatory claim.**

- selected paper baseline: `paper-0.5`
- best update-matched candidate: `gated-M1024-f0.70-u`
- advancement gate: **PASS**

## Aggregate ratios

| arm | ED2 / selected paper | ED2 / paper oracle | SW1 / selected paper |
|---|---:|---:|---:|
| gated-M1024-f0.70-u | 0.9059 | 0.9831 | 0.8754 |
| qld-v1 | 1.0489 | 1.1382 | 0.9271 |
| rsr-M1024-f0.70-u | 0.9446 | 1.0250 | 0.8863 |

## Advancement checks

- PASS: `unequal_ed2_improves_vs_qld_v1`
- PASS: `no_control_family_loses_more_than_5pct_vs_qld_v1`

## Best candidate families: gated-M1024-f0.70-u

| family | vs selected paper | vs QLD-v1 |
|---|---:|---:|
| control-connected | 1.1147 | 1.0000 |
| control-contaminated | 1.1619 | 0.9682 |
| control-equal | 0.7718 | 1.0000 |
| control-heavy-tail | 0.9637 | 1.0000 |
| control-heteroscedastic | 0.5374 | 0.9032 |
| control-overlap | 0.9269 | 1.0000 |
| unequal | 0.9276 | 0.7628 |

## Interpretation rule

N1 advances only when virtual batching improves the unequal family relative to QLD-v1 without making any control family more than 5% worse. N2 advances only at ED2 ratios <= .82 against the selected paper baseline and <= .95 against the per-cell paper oracle.
