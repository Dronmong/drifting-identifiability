# Persistent Quantile Transport: development results

**Profile:** `standard`  
**Status:** mutable mechanism development, not confirmatory evidence

| arm | ED2 / paper | ED2 / QLD | ED2 / LB-QCD | SW1 / LB-QCD | cell wins | gate |
|---|---:|---:|---:|---:|---:|---|
| gated-pqt-M1024 | 0.4326 | 0.5135 | 0.5443 | 0.5854 | 91.7% | PASS |
| pqt-B128 | 0.6513 | 0.7731 | 0.8194 | 0.8398 | 70.8% | PASS |

## Family ED2 ratios versus LB-QCD

### gated-pqt-M1024

- `control-connected`: `0.9857`
- `control-contaminated`: `0.2071`
- `control-equal`: `0.8534`
- `control-heavy-tail`: `0.7732`
- `control-heteroscedastic`: `0.7283`
- `control-overlap`: `0.4943`
- `unequal`: `0.4905`

### pqt-B128

- `control-connected`: `0.9857`
- `control-contaminated`: `0.5942`
- `control-equal`: `0.8534`
- `control-heavy-tail`: `0.7732`
- `control-heteroscedastic`: `1.0407`
- `control-overlap`: `0.4943`
- `unequal`: `0.8788`

## Interpretation guardrail

PQT is a one-dimensional nonparametric monotone generator. A favorable result tests persistent transport coordinates; it is not evidence for higher-dimensional or image generation. The matched `B128` arm is the architecture test. An improvement confined to gated `M1024` is a target-resolution result.

Candidate trial rows: `384`.
