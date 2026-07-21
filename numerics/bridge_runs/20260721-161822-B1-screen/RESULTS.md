# Calibrated bridge B1 (screen)

This is an exposed Registry-A mechanism test, not confirmation.

Gate pass: **False**

Decision: optimizer calibration fails; stop before scale stage

| arm | ED2 / QLD-v1 | SW1 / QLD-v1 |
|---|---:|---:|
| `bridge-calibrated` | 0.999703 | 0.990563 |
| `bridge-carry-copy` | 0.984216 | 0.999656 |
| `bridge-reset-full-lr` | 0.999588 | 0.963648 |
| `bridge-reset-quarter` | 0.996843 | 0.985643 |
| `bridge-warm-quarter` | 0.999696 | 0.990563 |
| `qld-full` | 1.1835 | 1.08686 |
| `qld-v1` | 1 | 1 |

## Registered gates

- `ed2_vs_qld_at_most_0.98`: **False**
- `sw1_vs_qld_at_most_0.99`: **False**
- `worst_connected_ed2_at_most_1.05`: **True**
- `each_initialization_ed2_at_most_1.02`: **True**
- `no_divergence`: **True**
- `no_denominator_floor`: **True**
- `median_step_ratio_between_0.5_and_1.5`: **False**
- `p90_step_ratio_at_most_2`: **True**
- `paper_state_restoration_exact`: **True**
