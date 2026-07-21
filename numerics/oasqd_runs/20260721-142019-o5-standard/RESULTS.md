# OA-SQD O5 development results

**Development evidence only; this is not a confirmatory claim.**

- predeclared OA candidate: `oasqd-edge-stop-Q0.08-H25-P50-N2048-K8`
- candidate / paper ED2: `0.8448`
- candidate / QLD-v1 ED2: `1.0031`
- candidate / fixed LB-QCD ED2: `0.9966`
- stage gate: **FAIL**

## Advancement gate

- ed2_vs_qld_at_most_0.95: `False`
- ed2_vs_paper_at_most_0.78: `False`
- each_init_vs_qld_at_most_0.98: `False`
- worst_family_vs_qld_at_most_1.05: `True`
- event_time_vs_fixed_at_most_1: `False`
- generator_evals_vs_paper_at_most_3: `True`
- divergence_no_worse: `True`

## Candidate details

```json
{
  "ed2_bootstrap_ci_vs_paper": [
    0.7356180807795055,
    0.934501397284545
  ],
  "ed2_bootstrap_ci_vs_qld": [
    0.990410628545211,
    1.019840210950453
  ],
  "cell_win_fraction_vs_paper": 0.6875,
  "sw1_ratio_vs_paper": 0.8668797357332961,
  "family_ed2_ratios_vs_qld": {
    "connected": 1.0,
    "contaminated": 1.0,
    "equal": 1.0119942110971738,
    "heavy-tail": 1.0,
    "heteroscedastic": 1.0,
    "overlap": 1.0,
    "resolution-boundary": 1.002955982666571,
    "unequal-interior": 0.9963834750794118,
    "unequal-tail": 1.0077448729113925
  },
  "init_ed2_ratios_vs_qld": {
    "concentrated": 1.0,
    "missing": 1.006163304387912
  },
  "event_time_ratio_vs_fixed": 1.0479894850374067,
  "generator_eval_ratio_vs_paper": 1.4435313098719371,
  "work_totals": {
    "fixed-lbqcd": {
      "wall_seconds": 772.2824031998753,
      "generator_example_evals": 249007104.0,
      "backward_example_evals": 136412160.0,
      "kernel_pairs": 3019898880.0,
      "total_new_target_samples": 137460736.0,
      "global_updates": 108360.0,
      "divergences": 0.0
    },
    "oa-stop-full-H25-P50-N2048-K8": {
      "wall_seconds": 693.0425545001053,
      "generator_example_evals": 178349952.0,
      "backward_example_evals": 96652160.0,
      "kernel_pairs": 3019898880.0,
      "total_new_target_samples": 98749312.0,
      "global_updates": 63985.0,
      "divergences": 0.0
    },
    "oasqd-edge-stop-Q0.08-H25-P50-N2048-K8": {
      "wall_seconds": 438.9422694998502,
      "generator_example_evals": 58636288.0,
      "backward_example_evals": 39321600.0,
      "kernel_pairs": 3019898880.0,
      "total_new_target_samples": 41265152.0,
      "global_updates": 1200.0,
      "divergences": 0.0
    },
    "paper-0.2": {
      "wall_seconds": 785.0694999999541,
      "generator_example_evals": 40521728.0,
      "backward_example_evals": 39321600.0,
      "kernel_pairs": 10066329600.0,
      "total_new_target_samples": 39321600.0,
      "global_updates": 0.0,
      "divergences": 0.0
    },
    "paper-0.5": {
      "wall_seconds": 769.9058757001767,
      "generator_example_evals": 40620032.0,
      "backward_example_evals": 39321600.0,
      "kernel_pairs": 10066329600.0,
      "total_new_target_samples": 39321600.0,
      "global_updates": 0.0,
      "divergences": 0.0
    },
    "paper-1": {
      "wall_seconds": 691.8973881002166,
      "generator_example_evals": 40896512.0,
      "backward_example_evals": 39321600.0,
      "kernel_pairs": 10066329600.0,
      "total_new_target_samples": 39321600.0,
      "global_updates": 0.0,
      "divergences": 0.0
    },
    "paper-2": {
      "wall_seconds": 704.6683601001569,
      "generator_example_evals": 41689088.0,
      "backward_example_evals": 39321600.0,
      "kernel_pairs": 10066329600.0,
      "total_new_target_samples": 39321600.0,
      "global_updates": 0.0,
      "divergences": 0.0
    },
    "paper-4": {
      "wall_seconds": 772.9849709999689,
      "generator_example_evals": 43560960.0,
      "backward_example_evals": 39321600.0,
      "kernel_pairs": 10066329600.0,
      "total_new_target_samples": 39321600.0,
      "global_updates": 0.0,
      "divergences": 0.0
    },
    "qld-v1": {
      "wall_seconds": 380.2123201000213,
      "generator_example_evals": 41054208.0,
      "backward_example_evals": 39321600.0,
      "kernel_pairs": 3019898880.0,
      "total_new_target_samples": 39321600.0,
      "global_updates": 0.0,
      "divergences": 0.0
    }
  }
}
```
