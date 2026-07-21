# Resolution-gated LB-QCD: confirmatory results

**Protocol:** `LBQCD-confirmatory-v1`  
**Verdict:** **FAIL**

## Primary results

- ED2 ratio versus selected paper: `0.8218`
- target-bootstrap ED2 95% CI: `[0.7649, 0.8915]`
- ED2 ratio versus per-cell paper oracle: `0.9437`
- cell win fraction: `0.9375`
- SW1 ratio versus selected paper: `0.8679`

## Gate

- FAIL: `ed2_ratio_at_most_0.80`
- PASS: `bootstrap_upper_below_1`
- PASS: `oracle_ed2_ratio_at_most_0.95`
- PASS: `cell_win_fraction_at_least_0.60`
- PASS: `all_family_ratios_at_most_1.10`
- PASS: `divergence_no_worse`

## Family ED2 ratios

- connected: `0.8944`
- contaminated: `0.8593`
- equal: `0.8642`
- heavy-tail: `0.7040`
- heteroscedastic: `0.8287`
- overlap: `0.6672`
- unequal: `0.8295`

## Cost

- summed worker wall ratio: `1.1226`
- generator-example-evaluation ratio: `6.5945`
- kernel-pair ratio: `0.3000`

## Scope

This is a frozen one-dimensional missing/concentrated-initialization test. It does not cover far starts, dimensions above one, or ImageNet-scale generation.
