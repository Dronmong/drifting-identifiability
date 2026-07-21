# OA-SQD O4 development results

**Development evidence only; this is not a confirmatory claim.**

- best OA candidate: `oasqd-stop-adaptive-H25-P50-N2048-K8`
- candidate / paper ED2: `1.2929`
- candidate / QLD-v1 ED2: `0.9948`
- candidate / fixed LB-QCD ED2: `1.0506`
- stage gate: **FAIL**

## Advancement gate


## Best-candidate details

```json
{
  "ed2_bootstrap_ci_vs_paper": [
    1.1715163411649818,
    1.4054666057891458
  ],
  "ed2_bootstrap_ci_vs_qld": [
    0.9916964678968752,
    1.0
  ],
  "cell_win_fraction_vs_paper": 0.0,
  "sw1_ratio_vs_paper": 1.126946046757328,
  "family_ed2_ratios_vs_qld": {
    "connected": 1.0,
    "equal": 0.9906526099651027,
    "unequal-tail": 0.9937874846817248
  },
  "init_ed2_ratios_vs_qld": {
    "concentrated": 0.9891755915803438,
    "missing": 1.0004680318158419
  },
  "event_time_ratio_vs_fixed": 1.0,
  "generator_eval_ratio_vs_paper": 4.125,
  "work_totals": {
    "fixed-lbqcd": {
      "wall_seconds": 0.5095930999959819,
      "generator_example_evals": 235008.0,
      "backward_example_evals": 118784.0,
      "kernel_pairs": 147456.0,
      "total_new_target_samples": 143360.0,
      "global_updates": 112.0,
      "divergences": 0.0
    },
    "oa-stop-strat-H25-P50-N2048-K8": {
      "wall_seconds": 0.1695765999611467,
      "generator_example_evals": 47232.0,
      "backward_example_evals": 8832.0,
      "kernel_pairs": 147456.0,
      "total_new_target_samples": 31872.0,
      "global_updates": 12.0,
      "divergences": 0.0
    },
    "oasqd-stop-adaptive-H25-P50-N2048-K8": {
      "wall_seconds": 0.15008200000738725,
      "generator_example_evals": 38016.0,
      "backward_example_evals": 8832.0,
      "kernel_pairs": 147456.0,
      "total_new_target_samples": 19584.0,
      "global_updates": 12.0,
      "divergences": 0.0
    },
    "paper-0.5": {
      "wall_seconds": 0.08512979999068193,
      "generator_example_evals": 9216.0,
      "backward_example_evals": 7680.0,
      "kernel_pairs": 491520.0,
      "total_new_target_samples": 7680.0,
      "global_updates": 0.0,
      "divergences": 0.0
    },
    "qld-v1": {
      "wall_seconds": 0.07147699999040924,
      "generator_example_evals": 9216.0,
      "backward_example_evals": 7680.0,
      "kernel_pairs": 147456.0,
      "total_new_target_samples": 7680.0,
      "global_updates": 0.0,
      "divergences": 0.0
    }
  }
}
```
