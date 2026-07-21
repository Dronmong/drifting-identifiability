# LB-QCD N2 development results

**Development evidence only. This registry is not sealed and these results are not a confirmatory claim.**

- selected paper baseline: `paper-0.5`
- best update-matched candidate: `gated-lookahead-M1024-f0.70-u`
- advancement gate: **FAIL**

## Aggregate ratios

| arm | ED2 / selected paper | ED2 / paper oracle | SW1 / selected paper |
|---|---:|---:|---:|
| gated-M1024-f0.70-u | 0.9059 | 0.9831 | 0.8754 |
| gated-align-M1024-f0.70-u | 0.9779 | 1.0612 | 0.9078 |
| gated-lookahead-M1024-f0.70-u | 0.9350 | 1.0146 | 0.9229 |
| lbqcd-align-M1024-f0.70-u | 0.9891 | 1.0733 | 0.9050 |
| lbqcd-lookahead-M1024-f0.70-u | 0.9435 | 1.0238 | 0.9205 |
| qld-v1 | 1.0489 | 1.1382 | 0.9271 |
| rsr-M1024-f0.70-u | 0.9446 | 1.0250 | 0.8863 |

## Advancement checks

- FAIL: `ed2_ratio_vs_selected_paper_at_most_0.82`
- FAIL: `ed2_ratio_vs_paper_oracle_at_most_0.95`

## Best candidate families: gated-lookahead-M1024-f0.70-u

| family | vs selected paper | vs QLD-v1 |
|---|---:|---:|
| control-connected | 0.9928 | 0.8907 |
| control-contaminated | 1.1569 | 0.9641 |
| control-equal | 0.7150 | 0.9265 |
| control-heavy-tail | 1.0949 | 1.1362 |
| control-heteroscedastic | 0.6110 | 1.0268 |
| control-overlap | 0.9802 | 1.0575 |
| unequal | 0.9692 | 0.7971 |

## Interpretation rule

N1 advances only when virtual batching improves the unequal family relative to QLD-v1 without making any control family more than 5% worse. N2 advances only at ED2 ratios <= .82 against the selected paper baseline and <= .95 against the per-cell paper oracle.
