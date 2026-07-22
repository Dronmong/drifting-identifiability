# Apache projected KLL pre-confirmation audit

Status: engineering audit; no endpoint targets used.

- projected Apache KLL invariants: PASS
- official single-sided normalized rank error at k=128: `0.020518`
- exact fixed-seed replay: unavailable in Apache Python API; serialized trained-state replay is audited instead

## Rank audit
- Apache maximum observed rank error: `0.016719`
- Apache 95th percentile maximum error: `0.011518`
- Apache median maximum error: `0.006510`
- local fixed-capacity median maximum error: `0.009531`
- gates:
  - `counts`: **PASS**
  - `monotone`: **PASS**
  - `observed_support`: **PASS**
  - `serialized_replay`: **PASS**
  - `max_rank_error_le_3x_official`: **PASS**
  - `p95_rank_error_le_2x_official`: **PASS**
  - `bounded_state_at_100k`: **PASS**
- engineering audit: **PASS**

## Promotion decision

Use Apache DataSketches KLL k=128 for the quality arm. Preserve serialized trained states because its Python API does not expose a seed for randomized compactions. The local fixed-capacity compactor remains development history, not the promoted implementation.
