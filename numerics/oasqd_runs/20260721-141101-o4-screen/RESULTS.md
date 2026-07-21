# OA-SQD O4 development results

**Development evidence only; this is not a confirmatory claim.**

- best OA candidate: `oasqd-stop-adaptive-H25-P50-N2048-K8`
- candidate / paper ED2: `0.8467`
- candidate / QLD-v1 ED2: `0.9908`
- candidate / fixed LB-QCD ED2: `0.9558`
- stage gate: **FAIL**

## Advancement gate


## Best-candidate details

```json
{
  "ed2_bootstrap_ci_vs_paper": [
    0.7520162152775998,
    1.0132149610277768
  ],
  "ed2_bootstrap_ci_vs_qld": [
    0.9028638888584263,
    1.033747520458147
  ],
  "cell_win_fraction_vs_paper": 0.71875,
  "sw1_ratio_vs_paper": 0.9021328501257925,
  "family_ed2_ratios_vs_qld": {
    "connected": 1.0,
    "contaminated": 0.9753046594821488,
    "equal": 1.0591147734092354,
    "heavy-tail": 1.0,
    "heteroscedastic": 1.0,
    "overlap": 1.0,
    "resolution-boundary": 0.892318970893612,
    "unequal-interior": 0.9914514773393092,
    "unequal-tail": 1.0129875232052528
  },
  "init_ed2_ratios_vs_qld": {
    "concentrated": 1.0289804531862163,
    "missing": 0.9540118046534969
  },
  "event_time_ratio_vs_fixed": 0.9963260456818589,
  "generator_eval_ratio_vs_paper": 3.530071456927352,
  "work_totals": {
    "fixed-lbqcd": {
      "wall_seconds": 79.15175059993635,
      "generator_example_evals": 33178624.0,
      "backward_example_evals": 17960960.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 18354176.0,
      "global_updates": 14560.0,
      "divergences": 0.0
    },
    "oa-stop-strat-H25-P50-N2048-K8": {
      "wall_seconds": 72.114536299865,
      "generator_example_evals": 17527808.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 14531712.0,
      "global_updates": 9855.0,
      "divergences": 0.0
    },
    "oasqd-once-adaptive-H25-P50-N2048-K8": {
      "wall_seconds": 49.21406789991306,
      "generator_example_evals": 9235456.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 5407232.0,
      "global_updates": 2300.0,
      "divergences": 0.0
    },
    "oasqd-stop-adaptive-H25-P50-N2048-K8": {
      "wall_seconds": 70.23435239988612,
      "generator_example_evals": 18211328.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 4442112.0,
      "global_updates": 9840.0,
      "divergences": 0.0
    },
    "oasqd-stop-adaptive-iid-H25-P50-N2048-K8": {
      "wall_seconds": 72.95397180010332,
      "generator_example_evals": 18189312.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 15197952.0,
      "global_updates": 9810.0,
      "divergences": 0.0
    },
    "oasqd-stop-systematic-H25-P50-N2048-K8": {
      "wall_seconds": 70.15219569989131,
      "generator_example_evals": 17511424.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 4442112.0,
      "global_updates": 9840.0,
      "divergences": 0.0
    },
    "paper-0.5": {
      "wall_seconds": 65.00202229997376,
      "generator_example_evals": 5158912.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 1258291200.0,
      "total_new_target_samples": 4915200.0,
      "global_updates": 0.0,
      "divergences": 0.0
    },
    "qld-v1": {
      "wall_seconds": 32.79461549996631,
      "generator_example_evals": 5236736.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 4915200.0,
      "global_updates": 0.0,
      "divergences": 0.0
    }
  }
}
```
