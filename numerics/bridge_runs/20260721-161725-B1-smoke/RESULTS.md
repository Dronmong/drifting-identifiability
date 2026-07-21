# Calibrated bridge B1 (smoke)

This is an exposed Registry-A mechanism test, not confirmation.

Gate pass: **False**

Decision: optimizer calibration fails; stop before scale stage

| arm | ED2 / QLD-v1 | SW1 / QLD-v1 |
|---|---:|---:|
| `bridge-calibrated` | 0.974242 | 1.0059 |
| `bridge-carry-copy` | 1.12504 | 1.08083 |
| `bridge-reset-full-lr` | 0.866619 | 0.94485 |
| `bridge-reset-quarter` | 0.963677 | 0.997419 |
| `bridge-warm-quarter` | 0.973559 | 1.00586 |
| `qld-full` | 0.899495 | 0.912815 |
| `qld-v1` | 1 | 1 |

## Registered gates

- `ed2_vs_qld_at_most_0.98`: **True**
- `sw1_vs_qld_at_most_0.99`: **False**
- `worst_connected_ed2_at_most_1.05`: **False**
- `each_initialization_ed2_at_most_1.02`: **False**
- `no_divergence`: **True**
- `no_denominator_floor`: **True**
- `median_step_ratio_between_0.5_and_1.5`: **False**
- `p90_step_ratio_at_most_2`: **True**
- `paper_state_restoration_exact`: **True**
