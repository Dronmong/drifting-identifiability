# PSQT accumulator confirmation

Status: disposable structural smoke run.

## Preflight
- Apache KLL invariants: PASS
- accumulator invariants: PASS
- target-only registry validation: PASS
- registry SHA-256: `1456da280b1eb13f00fecec517fa695cd40849dffea84067fee4d7a834d9f095`
- completed target 1/8: CF-F1-GMM-00
- completed target 2/8: CF-F2-ELLIPSES-00
- completed target 3/8: CF-F3-RARE-02-00
- completed target 4/8: CF-F4-CORRELATED-00
- completed target 5/8: CF-F5-PERTURBED-RING-00
- completed target 6/8: CF-F6-MULTICURVE-00
- completed target 7/8: CF-F7-SKEW-00
- completed target 8/8: CF-F8-OFFAXIS-BINARY-00

## Target-level outcomes
- `kll_hist_ed2`: ratio `0.1932`, 95% CI `[0.1932, 0.1932]`, wins `100.0%`
- `kll_paper_ed2`: ratio `0.0077`, 95% CI `[0.0077, 0.0077]`, wins `100.0%`
- `res_hist_ed2`: ratio `0.1746`, 95% CI `[0.1746, 0.1746]`, wins `100.0%`
- `res_paper_ed2`: ratio `0.0070`, 95% CI `[0.0070, 0.0070]`, wins `100.0%`
- KLL gates:
  - `ed2_vs_historical`: **PASS**
  - `ed2_vs_paper`: **PASS**
  - `sw1_vs_both`: **PASS**
  - `target_wins`: **PASS**
  - `family_robustness`: **PASS**
  - `zero_divergence`: **PASS**
  - `rare_recovery`: **FAIL**
  - `bridge_occupancy`: **PASS**
- Reservoir gates:
  - `ed2_vs_historical`: **PASS**
  - `ed2_vs_paper`: **PASS**
  - `sw1_nonregression`: **PASS**
  - `target_wins`: **PASS**
  - `state`: **FAIL**
  - `wall_time`: **PASS**
  - `zero_divergence`: **PASS**
  - `rare_recovery`: **FAIL**
- KLL promotion: **FAIL**
- Reservoir promotion: **FAIL**
- general low-dimensional improvement claim: **NOT SUPPORTED**

## Interpretation boundary

This protocol concerns a nonparametric 2D particle generator under the declared sample and particle budgets. It is not an ImageNet, encoder-feature, neural-generator, or all-bandwidth paper claim.
