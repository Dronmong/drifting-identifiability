# Persistent Quantile Transport: development results

**Profile:** `standard`  
**Status:** mutable mechanism development, not confirmatory evidence

| arm | ED2 / paper | ED2 / QLD | ED2 / LB-QCD | SW1 / LB-QCD | cell wins | gate |
|---|---:|---:|---:|---:|---:|---|
| gated-pqt-M1024-f0.70 | 0.4913 | 0.5832 | 0.6181 | 0.6622 | 79.2% | PASS |
| pqt-B128 | 0.6517 | 0.7736 | 0.8199 | 0.8392 | 70.8% | PASS |

## Family ED2 ratios versus LB-QCD

### gated-pqt-M1024-f0.70

- `control-connected`: `0.9856`
- `control-contaminated`: `0.2941`
- `control-equal`: `0.8538`
- `control-heavy-tail`: `0.7736`
- `control-heteroscedastic`: `0.8115`
- `control-overlap`: `0.4947`
- `unequal`: `0.5860`

### pqt-B128

- `control-connected`: `0.9856`
- `control-contaminated`: `0.5958`
- `control-equal`: `0.8538`
- `control-heavy-tail`: `0.7736`
- `control-heteroscedastic`: `1.0412`
- `control-overlap`: `0.4947`
- `unequal`: `0.8793`

## Interpretation guardrail

PQT is a one-dimensional nonparametric monotone generator. A favorable result tests persistent transport coordinates; it is not evidence for higher-dimensional or image generation. The matched `B128` arm is the architecture test. An improvement confined to gated `M1024` is a target-resolution result.

Candidate trial rows: `384`.
