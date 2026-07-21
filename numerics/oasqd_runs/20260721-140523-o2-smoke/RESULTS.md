# OA-SQD O2 development results

**Development evidence only; this is not a confirmatory claim.**

- best OA candidate: `oa-pulse-full-H25-P25-N2048-K8`
- candidate / paper ED2: `1.0709`
- candidate / QLD-v1 ED2: `1.0890`
- candidate / fixed LB-QCD ED2: `0.9628`
- stage gate: **PASS**

## Advancement gate

- endpoint_improves_vs_fixed: `True`
- fair_event_time_no_worse_than_fixed: `True`
- uses_fewer_global_updates_than_fixed: `True`

## Best-candidate details

```json
{
  "ed2_bootstrap_ci_vs_paper": [
    0.9789316417457622,
    1.1715163411649818
  ],
  "ed2_bootstrap_ci_vs_qld": [
    1.0,
    1.1858346978688918
  ],
  "cell_win_fraction_vs_paper": 0.25,
  "sw1_ratio_vs_paper": 1.083630951048298,
  "family_ed2_ratios_vs_qld": {
    "connected": 1.0,
    "unequal-tail": 1.1858346978688918
  },
  "init_ed2_ratios_vs_qld": {
    "concentrated": 1.1864728828441184,
    "missing": 0.9994621158355538
  },
  "event_time_ratio_vs_fixed": 1.0,
  "generator_eval_ratio_vs_paper": 5.635416666666667,
  "work_totals": {
    "fixed-lbqcd": {
      "wall_seconds": 0.22050049997051246,
      "generator_example_evals": 119040.0,
      "backward_example_evals": 60672.0,
      "kernel_pairs": 98304.0,
      "total_new_target_samples": 77056.0,
      "global_updates": 0.0,
      "divergences": 0.0
    },
    "oa-pulse-full-H25-P25-N2048-K8": {
      "wall_seconds": 0.09204220000538044,
      "generator_example_evals": 34624.0,
      "backward_example_evals": 11072.0,
      "kernel_pairs": 98304.0,
      "total_new_target_samples": 19264.0,
      "global_updates": 6.0,
      "divergences": 0.0
    },
    "oa-stop-full-H25-P25-N2048-K8": {
      "wall_seconds": 0.09103140002116561,
      "generator_example_evals": 34624.0,
      "backward_example_evals": 11072.0,
      "kernel_pairs": 98304.0,
      "total_new_target_samples": 19264.0,
      "global_updates": 6.0,
      "divergences": 0.0
    },
    "paper-0.5": {
      "wall_seconds": 0.05126919999020174,
      "generator_example_evals": 6144.0,
      "backward_example_evals": 5120.0,
      "kernel_pairs": 327680.0,
      "total_new_target_samples": 5120.0,
      "global_updates": 0.0,
      "divergences": 0.0
    },
    "qld-v1": {
      "wall_seconds": 0.04142870003124699,
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
