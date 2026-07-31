# Persistent Quantile Transport: development results

**Profile:** `standard`  
**Status:** mutable mechanism development, not confirmatory evidence

| arm | ED2 / paper | ED2 / QLD | ED2 / LB-QCD | SW1 / LB-QCD | cell wins | gate |
|---|---:|---:|---:|---:|---:|---|
| gated-pqt-M1024-f0.70 | 0.4902 | 0.5819 | 0.6167 | 0.6614 | 79.2% | PASS |

## Family ED2 ratios versus LB-QCD

### gated-pqt-M1024-f0.70

- `control-connected`: `0.9857`
- `control-contaminated`: `0.2924`
- `control-equal`: `0.8534`
- `control-heavy-tail`: `0.7732`
- `control-heteroscedastic`: `0.8095`
- `control-overlap`: `0.4943`
- `unequal`: `0.5843`

## Interpretation guardrail

PQT is a one-dimensional nonparametric monotone generator. A favorable result tests persistent transport coordinates; it is not evidence for higher-dimensional or image generation. The matched `B128` arm is the architecture test. An improvement confined to gated `M1024` is a target-resolution result.

Candidate trial rows: `192`.
