# OA-SQD O3 development results

**Development evidence only; this is not a confirmatory claim.**

- best OA candidate: `oa-stop-full-H25-P50-N2048-K8`
- candidate / paper ED2: `0.8534`
- candidate / QLD-v1 ED2: `0.9986`
- candidate / fixed LB-QCD ED2: `0.9633`
- stage gate: **FAIL**

## Advancement gate


## Best-candidate details

```json
{
  "ed2_bootstrap_ci_vs_paper": [
    0.7735440327422329,
    1.0233813148653719
  ],
  "ed2_bootstrap_ci_vs_qld": [
    0.9150974881507488,
    1.0499044578003562
  ],
  "cell_win_fraction_vs_paper": 0.71875,
  "sw1_ratio_vs_paper": 0.891592062885193,
  "family_ed2_ratios_vs_qld": {
    "connected": 1.0,
    "contaminated": 0.8612097701246628,
    "equal": 0.944209717318587,
    "heavy-tail": 1.0,
    "heteroscedastic": 1.0,
    "overlap": 1.0,
    "resolution-boundary": 0.9702060337582055,
    "unequal-interior": 1.0122311904733428,
    "unequal-tail": 1.0449823588111506
  },
  "init_ed2_ratios_vs_qld": {
    "concentrated": 1.0062127952581141,
    "missing": 0.9909719506503
  },
  "event_time_ratio_vs_fixed": 0.9963260456818589,
  "generator_eval_ratio_vs_paper": 5.1106589916633585,
  "work_totals": {
    "fixed-lbqcd": {
      "wall_seconds": 77.93046280005365,
      "generator_example_evals": 33178624.0,
      "backward_example_evals": 17960960.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 18354176.0,
      "global_updates": 14560.0,
      "divergences": 0.0
    },
    "oa-once-full-H25-P10-N2048-K8": {
      "wall_seconds": 43.51035919995047,
      "generator_example_evals": 7538176.0,
      "backward_example_evals": 5327360.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 6113792.0,
      "global_updates": 460.0,
      "divergences": 0.0
    },
    "oa-once-full-H25-P50-N2048-K8": {
      "wall_seconds": 49.73150799976429,
      "generator_example_evals": 11161088.0,
      "backward_example_evals": 6976000.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 7762432.0,
      "global_updates": 2300.0,
      "divergences": 0.0
    },
    "oa-once-strat-H25-P10-N2048-K8": {
      "wall_seconds": 44.52320779999718,
      "generator_example_evals": 7129088.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 6113792.0,
      "global_updates": 460.0,
      "divergences": 0.0
    },
    "oa-once-strat-H25-P50-N2048-K8": {
      "wall_seconds": 47.79336570011219,
      "generator_example_evals": 9102336.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 7762432.0,
      "global_updates": 2300.0,
      "divergences": 0.0
    },
    "oa-stop-full-H25-P50-N2048-K8": {
      "wall_seconds": 73.06185039991396,
      "generator_example_evals": 26365440.0,
      "backward_example_evals": 13749760.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 14536192.0,
      "global_updates": 9860.0,
      "divergences": 0.0
    },
    "oa-stop-strat-H25-P50-N2048-K8": {
      "wall_seconds": 72.17074979981408,
      "generator_example_evals": 17527808.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 14531712.0,
      "global_updates": 9855.0,
      "divergences": 0.0
    },
    "paper-0.5": {
      "wall_seconds": 66.0712887001282,
      "generator_example_evals": 5158912.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 1258291200.0,
      "total_new_target_samples": 4915200.0,
      "global_updates": 0.0,
      "divergences": 0.0
    },
    "qld-v1": {
      "wall_seconds": 31.457827399892267,
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
