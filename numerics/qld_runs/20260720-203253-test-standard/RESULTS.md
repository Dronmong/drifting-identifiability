# QLD confirmatory results

**Gate: FAIL**

- selected paper tau: `0.5`
- primary ED2 ratio: `0.9105`
- paired target-bootstrap 95% CI: `[0.8270, 0.9873]`
- cell win fraction: `0.719`
- secondary SW1 ratio: `0.8949`
- oracle-per-target paper ED2 ratio: `0.9774`

## Gate checks

- FAIL: `ratio_at_most_0.80`
- PASS: `bootstrap_upper_below_1`
- PASS: `win_fraction_at_least_0.60`
- PASS: `all_family_ratios_at_most_1.10`
- PASS: `divergence_no_worse`

## Family ED2 ratios

- connected-heavy-tail: `0.9635`
- contaminated: `0.8113`
- equal: `0.8945`
- heteroscedastic: `0.8687`
- overlap: `0.7469`
- unequal: `1.0726`

## Scope

This gate covers only the sealed one-dimensional fission suite and its missing/concentrated initializations. It is not an ImageNet, high-dimensional, or arbitrary-start claim.
