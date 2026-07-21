# OA-SQD O3 development results

**Development evidence only; this is not a confirmatory claim.**

- best OA candidate: `oa-once-full-H25-P10-N2048-K8`
- candidate / paper ED2: `1.3304`
- candidate / QLD-v1 ED2: `0.9988`
- candidate / fixed LB-QCD ED2: `1.0158`
- stage gate: **FAIL**

## Advancement gate


## Best-candidate details

```json
{
  "ed2_bootstrap_ci_vs_paper": [
    1.1715163411649818,
    1.5107465777126292
  ],
  "ed2_bootstrap_ci_vs_qld": [
    0.9975178973211952,
    1.0
  ],
  "cell_win_fraction_vs_paper": 0.0,
  "sw1_ratio_vs_paper": 1.1705537825700507,
  "family_ed2_ratios_vs_qld": {
    "connected": 1.0,
    "unequal-tail": 0.9975178973211952
  },
  "init_ed2_ratios_vs_qld": {
    "concentrated": 0.9950507138312807,
    "missing": 1.0024794550223626
  },
  "event_time_ratio_vs_fixed": 1.0,
  "generator_eval_ratio_vs_paper": 5.635416666666667,
  "work_totals": {
    "fixed-lbqcd": {
      "wall_seconds": 0.25182700000004843,
      "generator_example_evals": 119040.0,
      "backward_example_evals": 60672.0,
      "kernel_pairs": 98304.0,
      "total_new_target_samples": 77056.0,
      "global_updates": 56.0,
      "divergences": 0.0
    },
    "oa-once-full-H25-P10-N2048-K8": {
      "wall_seconds": 0.10624730002018623,
      "generator_example_evals": 34624.0,
      "backward_example_evals": 11072.0,
      "kernel_pairs": 98304.0,
      "total_new_target_samples": 19264.0,
      "global_updates": 6.0,
      "divergences": 0.0
    },
    "oa-once-strat-H25-P10-N2048-K8": {
      "wall_seconds": 0.1011622000078205,
      "generator_example_evals": 29248.0,
      "backward_example_evals": 5696.0,
      "kernel_pairs": 98304.0,
      "total_new_target_samples": 19264.0,
      "global_updates": 6.0,
      "divergences": 0.0
    },
    "paper-0.5": {
      "wall_seconds": 0.04727780004031956,
      "generator_example_evals": 6144.0,
      "backward_example_evals": 5120.0,
      "kernel_pairs": 327680.0,
      "total_new_target_samples": 5120.0,
      "global_updates": 0.0,
      "divergences": 0.0
    },
    "qld-v1": {
      "wall_seconds": 0.03703449998283759,
      "generator_example_evals": 6144.0,
      "backward_example_evals": 5120.0,
      "kernel_pairs": 98304.0,
      "total_new_target_samples": 5120.0,
      "global_updates": 0.0,
      "divergences": 0.0
    }
  }
}
```
