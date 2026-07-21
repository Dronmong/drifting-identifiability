# LB-QCD N1 development results

**Development evidence only. This registry is not sealed and these results are not a confirmatory claim.**

- selected paper baseline: `paper-0.5`
- best update-matched candidate: `gated-M1024-f0.70-u`
- advancement gate: **PASS**

## Aggregate ratios

| arm | ED2 / selected paper | ED2 / paper oracle | SW1 / selected paper |
|---|---:|---:|---:|
| gated-M1024-f0.70-u | 3.3089 | 3.3089 | 1.6102 |
| qld-v1 | 3.5331 | 3.5331 | 1.6989 |

## Advancement checks

- PASS: `unequal_ed2_improves_vs_qld_v1`
- PASS: `no_control_family_loses_more_than_5pct_vs_qld_v1`

## Best candidate families: gated-M1024-f0.70-u

| family | vs selected paper | vs QLD-v1 |
|---|---:|---:|
| control-heavy-tail | 2.0467 | 1.0000 |
| control-overlap | 1.3332 | 1.0000 |
| unequal | 13.2773 | 0.8215 |

## Uncertainty and cost

- target-bootstrap ED2 95% CI: `[0.8742, 8.9229]`
- target-bootstrap SW1 95% CI: `[0.7484, 3.1478]`
- cell win fraction: `0.0000`
- summed worker wall ratio: `1.0096`
- generator-example-evaluation ratio: `4.5000`
- kernel-pair ratio: `0.3000`
- divergences: `0`

## Interpretation rule

N1 advances only when virtual batching improves the unequal family relative to QLD-v1 without making any control family more than 5% worse. N2 advances only at ED2 ratios <= .82 against the selected paper baseline and <= .95 against the per-cell paper oracle.
