# Persistent Quantile Transport: development results

**Profile:** `screen`  
**Status:** mutable mechanism development, not confirmatory evidence

| arm | ED2 / paper | ED2 / QLD | ED2 / LB-QCD | SW1 / LB-QCD | cell wins | gate |
|---|---:|---:|---:|---:|---:|---|
| gated-pqt-M1024 | 0.5890 | 0.5615 | 0.6502 | 0.6456 | 79.2% | FAIL |
| pqt-B128 | 0.7855 | 0.7488 | 0.8670 | 0.8612 | 66.7% | FAIL |

## Family ED2 ratios versus LB-QCD

### gated-pqt-M1024

- `control-connected`: `0.6818`
- `control-contaminated`: `0.2813`
- `control-equal`: `0.9411`
- `control-heavy-tail`: `0.7846`
- `control-heteroscedastic`: `1.2178`
- `control-overlap`: `1.0815`
- `unequal`: `0.5592`

### pqt-B128

- `control-connected`: `0.6818`
- `control-contaminated`: `0.6022`
- `control-equal`: `0.9411`
- `control-heavy-tail`: `0.7846`
- `control-heteroscedastic`: `1.3266`
- `control-overlap`: `1.0815`
- `unequal`: `0.8636`

## Interpretation guardrail

PQT is a one-dimensional nonparametric monotone generator. A favorable result tests persistent transport coordinates; it is not evidence for higher-dimensional or image generation. The matched `B128` arm is the architecture test. An improvement confined to gated `M1024` is a target-resolution result.

Candidate trial rows: `144`.
