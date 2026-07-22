# Persistent Quantile Transport: frozen confirmatory results

**Protocol:** `PQT-confirmatory-v1`  
**Verdict:** **PASS**

## Primary comparisons

| comparison | ED2 ratio | 95% interval | SW1 ratio | 95% interval |
|---|---:|---:|---:|---:|
| PQT / selected paper | 0.3487 | `[0.2223,0.4940]` | 0.4582 | `[0.3495,0.5684]` |
| PQT / QLD-v1 | 0.4036 | `[0.3064,0.5240]` | 0.5482 | `[0.4715,0.6281]` |
| PQT / gated LB-QCD | 0.4230 | `[0.3258,0.5344]` | 0.5602 | `[0.4872,0.6360]` |

ED2 ratio versus the per-cell paper oracle: `0.5183`.

## Conjunctive gate

- PASS: `ed2_vs_lbqcd_at_most_0.80`
- PASS: `ed2_bootstrap_upper_vs_lbqcd_below_1`
- PASS: `sw1_vs_lbqcd_at_most_0.85`
- PASS: `sw1_bootstrap_upper_vs_lbqcd_below_1`
- PASS: `ed2_vs_selected_paper_at_most_0.70`
- PASS: `ed2_vs_paper_oracle_at_most_0.85`
- PASS: `cell_win_fraction_vs_lbqcd_at_least_0.70`
- PASS: `all_family_ed2_ratios_vs_lbqcd_at_most_1.10`
- PASS: `all_init_ed2_ratios_vs_lbqcd_below_1`
- PASS: `far_ed2_ratio_vs_lbqcd_at_most_0.80`
- PASS: `divergence_no_worse_than_lbqcd`
- PASS: `target_sample_budget_equal`
- PASS: `routing_decisions_equal`

## Family ED2 ratios versus LB-QCD

- `connected`: `0.2789`
- `contaminated`: `0.1548`
- `equal`: `0.6323`
- `heavy-tail`: `0.5694`
- `heteroscedastic`: `0.4388`
- `overlap`: `0.6068`
- `unequal`: `0.4158`

## Initialization ED2 ratios versus LB-QCD

- `concentrated`: `0.6076`
- `far`: `0.2160`
- `missing`: `0.5767`

## Scope

This verdict concerns the predeclared one-dimensional synthetic registry and its three initialization regimes. It does not establish higher-dimensional, encoder-feature, or image-generation superiority.
