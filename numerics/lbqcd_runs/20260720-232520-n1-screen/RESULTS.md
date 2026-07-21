# LB-QCD N1 development results

**Development evidence only. This registry is not sealed and these results are not a confirmatory claim.**

- selected paper baseline: `paper-0.5`
- best update-matched candidate: `rsr-M1024-f0.70-u`
- advancement gate: **FAIL**

## Aggregate ratios

| arm | ED2 / selected paper | ED2 / paper oracle | SW1 / selected paper |
|---|---:|---:|---:|
| qld-v1 | 1.0489 | 1.1382 | 0.9271 |
| rsr-M1024-f0.70-e | 22.6572 | 24.5860 | 6.5049 |
| rsr-M1024-f0.70-u | 0.9446 | 1.0250 | 0.8863 |
| rsr-M512-f0.70-e | 6.3232 | 6.8614 | 3.0164 |
| rsr-M512-f0.70-u | 0.9526 | 1.0337 | 0.9267 |

## Advancement checks

- PASS: `unequal_ed2_improves_vs_qld_v1`
- FAIL: `no_control_family_loses_more_than_5pct_vs_qld_v1`

## Best candidate families: rsr-M1024-f0.70-u

| family | vs selected paper | vs QLD-v1 |
|---|---:|---:|
| control-connected | 1.0281 | 0.9223 |
| control-contaminated | 1.1619 | 0.9682 |
| control-equal | 0.6902 | 0.8943 |
| control-heavy-tail | 1.1847 | 1.2293 |
| control-heteroscedastic | 0.5925 | 0.9958 |
| control-overlap | 1.3695 | 1.4775 |
| unequal | 0.9276 | 0.7628 |

## Interpretation rule

N1 advances only when virtual batching improves the unequal family relative to QLD-v1 without making any control family more than 5% worse. N2 advances only at ED2 ratios <= .82 against the selected paper baseline and <= .95 against the per-cell paper oracle.
