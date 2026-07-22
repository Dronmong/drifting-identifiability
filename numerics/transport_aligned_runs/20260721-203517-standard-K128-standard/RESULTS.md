# Persistent Quantile Transport: development results

**Profile:** `standard`  
**Status:** mutable mechanism development, not confirmatory evidence

| arm | ED2 / paper | ED2 / QLD | ED2 / LB-QCD | SW1 / LB-QCD | cell wins | gate |
|---|---:|---:|---:|---:|---:|---|
| gated-pqt-M1024-f0.70 | 0.4952 | 0.5878 | 0.6231 | 0.6657 | 79.2% | PASS |
| pqt-B128 | 0.6536 | 0.7758 | 0.8223 | 0.8383 | 70.8% | PASS |

## Family ED2 ratios versus LB-QCD

### gated-pqt-M1024-f0.70

- `control-connected`: `0.9863`
- `control-contaminated`: `0.3009`
- `control-equal`: `0.8556`
- `control-heavy-tail`: `0.7765`
- `control-heteroscedastic`: `0.8171`
- `control-overlap`: `0.4970`
- `unequal`: `0.5913`

### pqt-B128

- `control-connected`: `0.9863`
- `control-contaminated`: `0.6011`
- `control-equal`: `0.8556`
- `control-heavy-tail`: `0.7765`
- `control-heteroscedastic`: `1.0427`
- `control-overlap`: `0.4970`
- `unequal`: `0.8811`

## Interpretation guardrail

PQT is a one-dimensional nonparametric monotone generator. A favorable result tests persistent transport coordinates; it is not evidence for higher-dimensional or image generation. The matched `B128` arm is the architecture test. An improvement confined to gated `M1024` is a target-resolution result.

Candidate trial rows: `384`.
