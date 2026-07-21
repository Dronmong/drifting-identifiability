# OA-SQD O2 development results

**Development evidence only; this is not a confirmatory claim.**

- best OA candidate: `oa-once-full-H25-P50-N2048-K8`
- candidate / paper ED2: `0.8601`
- candidate / QLD-v1 ED2: `1.0064`
- candidate / fixed LB-QCD ED2: `0.9708`
- stage gate: **FAIL**

## Advancement gate

- endpoint_improves_vs_fixed: `True`
- fair_event_time_no_worse_than_fixed: `False`
- uses_fewer_global_updates_than_fixed: `True`

## Best-candidate details

```json
{
  "ed2_bootstrap_ci_vs_paper": [
    0.7548735939875263,
    1.046740871843448
  ],
  "ed2_bootstrap_ci_vs_qld": [
    0.8927818476804336,
    1.0853504029306087
  ],
  "cell_win_fraction_vs_paper": 0.6875,
  "sw1_ratio_vs_paper": 0.8982340669405074,
  "family_ed2_ratios_vs_qld": {
    "connected": 1.0,
    "contaminated": 0.8458838326700977,
    "equal": 0.8116310891699153,
    "heavy-tail": 1.0,
    "heteroscedastic": 1.0,
    "overlap": 1.0,
    "resolution-boundary": 1.0200303633742522,
    "unequal-interior": 1.0122311904733428,
    "unequal-tail": 1.0863961321519526
  },
  "init_ed2_ratios_vs_qld": {
    "concentrated": 1.0397509129374247,
    "missing": 0.9741373605415933
  },
  "event_time_ratio_vs_fixed": 1.0147960064359274,
  "generator_eval_ratio_vs_paper": 2.1634577213179833,
  "work_totals": {
    "fixed-lbqcd": {
      "wall_seconds": 75.6698043999786,
      "generator_example_evals": 33178624.0,
      "backward_example_evals": 17960960.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 18354176.0,
      "global_updates": 14560.0,
      "divergences": 0.0
    },
    "oa-once-full-H25-P10-N2048-K8": {
      "wall_seconds": 43.240785099886125,
      "generator_example_evals": 7538176.0,
      "backward_example_evals": 5327360.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 6113792.0,
      "global_updates": 460.0,
      "divergences": 0.0
    },
    "oa-once-full-H25-P25-N1024-K8": {
      "wall_seconds": 45.6225543001201,
      "generator_example_evals": 8157440.0,
      "backward_example_evals": 5945600.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 6732032.0,
      "global_updates": 1150.0,
      "divergences": 0.0
    },
    "oa-once-full-H25-P25-N2048-K4": {
      "wall_seconds": 45.2318593999953,
      "generator_example_evals": 8370688.0,
      "backward_example_evals": 5542400.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 6328832.0,
      "global_updates": 700.0,
      "divergences": 0.0
    },
    "oa-once-full-H25-P25-N2048-K8": {
      "wall_seconds": 46.67893690013443,
      "generator_example_evals": 8866048.0,
      "backward_example_evals": 5945600.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 6732032.0,
      "global_updates": 1150.0,
      "divergences": 0.0
    },
    "oa-once-full-H25-P50-N2048-K8": {
      "wall_seconds": 49.13986090014805,
      "generator_example_evals": 11161088.0,
      "backward_example_evals": 6976000.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 7762432.0,
      "global_updates": 2300.0,
      "divergences": 0.0
    },
    "oa-once-full-H50-P25-N2048-K8": {
      "wall_seconds": 44.36990349998814,
      "generator_example_evals": 8087296.0,
      "backward_example_evals": 5856000.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 6642432.0,
      "global_updates": 1050.0,
      "divergences": 0.0
    },
    "paper-0.5": {
      "wall_seconds": 65.83862369990675,
      "generator_example_evals": 5158912.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 1258291200.0,
      "total_new_target_samples": 4915200.0,
      "global_updates": 0.0,
      "divergences": 0.0
    },
    "qld-v1": {
      "wall_seconds": 32.41001700007473,
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
