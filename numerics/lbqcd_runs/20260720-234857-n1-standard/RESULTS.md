# LB-QCD N1 development results

**Development evidence only. This registry is not sealed and these results are not a confirmatory claim.**

- selected paper baseline: `paper-0.5`
- best update-matched candidate: `gated-M1024-f0.70-u`
- advancement gate: **PASS**

## Aggregate ratios

| arm | ED2 / selected paper | ED2 / paper oracle | SW1 / selected paper |
|---|---:|---:|---:|
| gated-M1024-f0.70-u | 0.7948 | 0.8685 | 0.8456 |
| qld-v1 | 0.8424 | 0.9206 | 0.8540 |

## Advancement checks

- PASS: `unequal_ed2_improves_vs_qld_v1`
- PASS: `no_control_family_loses_more_than_5pct_vs_qld_v1`

## Best candidate families: gated-M1024-f0.70-u

| family | vs selected paper | vs QLD-v1 |
|---|---:|---:|
| control-connected | 0.7246 | 1.0000 |
| control-contaminated | 0.8877 | 0.8793 |
| control-equal | 0.8267 | 1.0000 |
| control-heavy-tail | 0.6824 | 1.0000 |
| control-heteroscedastic | 0.6251 | 0.8447 |
| control-overlap | 0.8358 | 1.0000 |
| unequal | 0.8335 | 0.9353 |

## Uncertainty and cost

- target-bootstrap ED2 95% CI: `[0.7063, 0.8910]`
- target-bootstrap SW1 95% CI: `[0.7679, 0.9229]`
- cell win fraction: `0.7917`
- summed worker wall ratio: `1.2217`
- generator-example-evaluation ratio: `7.8906`
- kernel-pair ratio: `0.3000`
- divergences: `0`

## Interpretation rule

N1 advances only when virtual batching improves the unequal family relative to QLD-v1 without making any control family more than 5% worse. N2 advances only at ED2 ratios <= .82 against the selected paper baseline and <= .95 against the per-cell paper oracle.
