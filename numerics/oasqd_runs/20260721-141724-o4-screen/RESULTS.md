# OA-SQD O4 development results

**Development evidence only; this is not a confirmatory claim.**

- best OA candidate: `oasqd-edge-stop-Q0.08-H25-P50-N2048-K8`
- candidate / paper ED2: `0.8458`
- candidate / QLD-v1 ED2: `0.9896`
- candidate / fixed LB-QCD ED2: `0.9546`
- stage gate: **FAIL**

## Advancement gate


## Best-candidate details

```json
{
  "ed2_bootstrap_ci_vs_paper": [
    0.7547012056596037,
    1.0313930466661712
  ],
  "ed2_bootstrap_ci_vs_qld": [
    0.9405586903935509,
    1.0134553435137463
  ],
  "cell_win_fraction_vs_paper": 0.6875,
  "sw1_ratio_vs_paper": 0.9245514906774815,
  "family_ed2_ratios_vs_qld": {
    "connected": 1.0,
    "contaminated": 1.0,
    "equal": 0.7758518895769332,
    "heavy-tail": 1.0,
    "heteroscedastic": 1.0,
    "overlap": 1.0,
    "resolution-boundary": 1.0,
    "unequal-interior": 1.0,
    "unequal-tail": 1.0175384851372682
  },
  "init_ed2_ratios_vs_qld": {
    "concentrated": 1.0,
    "missing": 0.9793583495744096
  },
  "event_time_ratio_vs_fixed": 1.0537041978709456,
  "generator_eval_ratio_vs_paper": 1.5155815799920604,
  "work_totals": {
    "fixed-lbqcd": {
      "wall_seconds": 76.39015650001238,
      "generator_example_evals": 33178624.0,
      "backward_example_evals": 17960960.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 18354176.0,
      "global_updates": 14560.0,
      "divergences": 0.0
    },
    "oasqd-edge-stop-Q0.05-H25-P50-N2048-K8": {
      "wall_seconds": 46.545057599956635,
      "generator_example_evals": 7592448.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 5695232.0,
      "global_updates": 50.0,
      "divergences": 0.0
    },
    "oasqd-edge-stop-Q0.08-H25-P50-N2048-K8": {
      "wall_seconds": 46.50286779983435,
      "generator_example_evals": 7818752.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 5644032.0,
      "global_updates": 450.0,
      "divergences": 0.0
    },
    "oasqd-edge-stop-Q0.10-H25-P50-N2048-K8": {
      "wall_seconds": 47.04136420003488,
      "generator_example_evals": 7990784.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 5631232.0,
      "global_updates": 550.0,
      "divergences": 0.0
    },
    "oasqd-edge-stop-Q0.12-H25-P50-N2048-K8": {
      "wall_seconds": 48.05756420022226,
      "generator_example_evals": 8162304.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 5612032.0,
      "global_updates": 700.0,
      "divergences": 0.0
    },
    "oasqd-edge-stop-Q0.15-H25-P50-N2048-K8": {
      "wall_seconds": 47.16153309997753,
      "generator_example_evals": 8289792.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 5567232.0,
      "global_updates": 1050.0,
      "divergences": 0.0
    },
    "oasqd-edge-stop-Q0.18-H25-P50-N2048-K8": {
      "wall_seconds": 47.96169799999916,
      "generator_example_evals": 8384256.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 5557632.0,
      "global_updates": 1125.0,
      "divergences": 0.0
    },
    "oasqd-edge-stop-Q0.20-H25-P50-N2048-K8": {
      "wall_seconds": 48.00833709986182,
      "generator_example_evals": 8392704.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 5554432.0,
      "global_updates": 1150.0,
      "divergences": 0.0
    },
    "oasqd-stop-adaptive-H25-P50-N2048-K8": {
      "wall_seconds": 71.24038980016485,
      "generator_example_evals": 18211328.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 377487360.0,
      "total_new_target_samples": 4442112.0,
      "global_updates": 9840.0,
      "divergences": 0.0
    },
    "paper-0.5": {
      "wall_seconds": 68.92136470010155,
      "generator_example_evals": 5158912.0,
      "backward_example_evals": 4915200.0,
      "kernel_pairs": 1258291200.0,
      "total_new_target_samples": 4915200.0,
      "global_updates": 0.0,
      "divergences": 0.0
    },
    "qld-v1": {
      "wall_seconds": 33.89476279990049,
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
