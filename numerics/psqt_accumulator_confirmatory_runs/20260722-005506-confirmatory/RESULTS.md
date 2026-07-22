# PSQT accumulator confirmation

Status: **sealed confirmatory execution**.

## Preflight
- Apache KLL invariants: PASS
- accumulator invariants: PASS
- target-only registry validation: PASS
- registry SHA-256: `1f8196b8dc73acddec03bc6ed31f6c19c3d9236a321f84e1ed5df459ad867a54`
- completed target 1/64: CF-F1-GMM-00
- completed target 2/64: CF-F1-GMM-01
- completed target 3/64: CF-F1-GMM-02
- completed target 4/64: CF-F1-GMM-03
- completed target 5/64: CF-F1-GMM-04
- completed target 6/64: CF-F1-GMM-05
- completed target 7/64: CF-F1-GMM-06
- completed target 8/64: CF-F1-GMM-07
- completed target 9/64: CF-F2-ELLIPSES-00
- completed target 10/64: CF-F2-ELLIPSES-01
- completed target 11/64: CF-F2-ELLIPSES-02
- completed target 12/64: CF-F2-ELLIPSES-03
- completed target 13/64: CF-F2-ELLIPSES-04
- completed target 14/64: CF-F2-ELLIPSES-05
- completed target 15/64: CF-F2-ELLIPSES-06
- completed target 16/64: CF-F2-ELLIPSES-07
- completed target 17/64: CF-F3-RARE-02-00
- completed target 18/64: CF-F3-RARE-05-01
- completed target 19/64: CF-F3-RARE-10-02
- completed target 20/64: CF-F3-RARE-02-03
- completed target 21/64: CF-F3-RARE-05-04
- completed target 22/64: CF-F3-RARE-10-05
- completed target 23/64: CF-F3-RARE-02-06
- completed target 24/64: CF-F3-RARE-05-07
- completed target 25/64: CF-F4-CORRELATED-00
- completed target 26/64: CF-F4-CORRELATED-01
- completed target 27/64: CF-F4-CORRELATED-02
- completed target 28/64: CF-F4-CORRELATED-03
- completed target 29/64: CF-F4-CORRELATED-04
- completed target 30/64: CF-F4-CORRELATED-05
- completed target 31/64: CF-F4-CORRELATED-06
- completed target 32/64: CF-F4-CORRELATED-07
- completed target 33/64: CF-F5-PERTURBED-RING-00
- completed target 34/64: CF-F5-SPIRAL-01
- completed target 35/64: CF-F5-ARC-02
- completed target 36/64: CF-F5-PERTURBED-RING-03
- completed target 37/64: CF-F5-SPIRAL-04
- completed target 38/64: CF-F5-ARC-05
- completed target 39/64: CF-F5-PERTURBED-RING-06
- completed target 40/64: CF-F5-SPIRAL-07
- completed target 41/64: CF-F6-MULTICURVE-00
- completed target 42/64: CF-F6-MULTICURVE-01
- completed target 43/64: CF-F6-MULTICURVE-02
- completed target 44/64: CF-F6-MULTICURVE-03
- completed target 45/64: CF-F6-MULTICURVE-04
- completed target 46/64: CF-F6-MULTICURVE-05
- completed target 47/64: CF-F6-MULTICURVE-06
- completed target 48/64: CF-F6-MULTICURVE-07
- completed target 49/64: CF-F7-SKEW-00
- completed target 50/64: CF-F7-SKEW-01
- completed target 51/64: CF-F7-SKEW-02
- completed target 52/64: CF-F7-SKEW-03
- completed target 53/64: CF-F7-SKEW-04
- completed target 54/64: CF-F7-SKEW-05
- completed target 55/64: CF-F7-SKEW-06
- completed target 56/64: CF-F7-SKEW-07
- completed target 57/64: CF-F8-OFFAXIS-BINARY-00
- completed target 58/64: CF-F8-CHECKERBOARD-01
- completed target 59/64: CF-F8-NONLINEAR-SINE-02
- completed target 60/64: CF-F8-OFFAXIS-BINARY-03
- completed target 61/64: CF-F8-CHECKERBOARD-04
- completed target 62/64: CF-F8-NONLINEAR-SINE-05
- completed target 63/64: CF-F8-OFFAXIS-BINARY-06
- completed target 64/64: CF-F8-CHECKERBOARD-07

## Target-level outcomes
- `kll_hist_ed2`: ratio `0.3370`, 95% CI `[0.3111, 0.3669]`, wins `98.4%`
- `kll_paper_ed2`: ratio `0.0762`, 95% CI `[0.0634, 0.0927]`, wins `100.0%`
- `res_hist_ed2`: ratio `0.5529`, 95% CI `[0.5108, 0.6018]`, wins `78.1%`
- `res_paper_ed2`: ratio `0.1250`, 95% CI `[0.1038, 0.1524]`, wins `100.0%`
- KLL gates:
  - `ed2_vs_historical`: **PASS**
  - `ed2_vs_paper`: **PASS**
  - `sw1_vs_both`: **PASS**
  - `target_wins`: **PASS**
  - `family_robustness`: **PASS**
  - `zero_divergence`: **PASS**
  - `rare_recovery`: **PASS**
  - `bridge_occupancy`: **PASS**
- Reservoir gates:
  - `ed2_vs_historical`: **PASS**
  - `ed2_vs_paper`: **PASS**
  - `sw1_nonregression`: **PASS**
  - `target_wins`: **PASS**
  - `state`: **PASS**
  - `wall_time`: **PASS**
  - `zero_divergence`: **PASS**
  - `rare_recovery`: **PASS**
- KLL promotion: **PASS**
- Reservoir promotion: **PASS**
- general low-dimensional improvement claim: **SUPPORTED**

## Interpretation boundary

This protocol concerns a nonparametric 2D particle generator under the declared sample and particle budgets. It is not an ImageNet, encoder-feature, neural-generator, or all-bandwidth paper claim.
