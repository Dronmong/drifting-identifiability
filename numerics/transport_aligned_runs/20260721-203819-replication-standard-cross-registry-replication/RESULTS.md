# Persistent Quantile Transport: development results

**Profile:** `standard`  
**Status:** mutable mechanism development, not confirmatory evidence

| arm | ED2 / paper | ED2 / QLD | ED2 / LB-QCD | SW1 / LB-QCD | cell wins | gate |
|---|---:|---:|---:|---:|---:|---|
| gated-pqt-M1024-f0.70 | 0.5281 | 0.6318 | 0.6427 | 0.7044 | 90.6% | PASS |

## Family ED2 ratios versus LB-QCD

### gated-pqt-M1024-f0.70

- `connected`: `0.9042`
- `contaminated`: `0.3372`
- `equal`: `0.7952`
- `heavy-tail`: `0.8066`
- `heteroscedastic`: `0.7774`
- `overlap`: `0.7006`
- `unequal`: `0.5548`

## Interpretation guardrail

PQT is a one-dimensional nonparametric monotone generator. A favorable result tests persistent transport coordinates; it is not evidence for higher-dimensional or image generation. The matched `B128` arm is the architecture test. An improvement confined to gated `M1024` is a target-resolution result.

Candidate trial rows: `640`.
