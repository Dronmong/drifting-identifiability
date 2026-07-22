# PSQT target-quantile accumulator repair

Status: exploratory development; not frozen confirmation.

## Invariants
- accumulator invariants: PASS
- historical PSQT invariants: PASS
- completed paired cell 1/10: PS2-GMM4-equal/concentrated/seed0
- completed paired cell 2/10: PS2-GMM4-equal/far/seed0
- completed paired cell 3/10: PS2-diagonal-dependence/concentrated/seed0
- completed paired cell 4/10: PS2-diagonal-dependence/far/seed0
- completed paired cell 5/10: PS2-skew/concentrated/seed0
- completed paired cell 6/10: PS2-skew/far/seed0
- completed paired cell 7/10: PS2-diagonal-minority-05/concentrated/seed0
- completed paired cell 8/10: PS2-diagonal-minority-05/far/seed0
- completed paired cell 9/10: PS2-diagonal-minority-10/concentrated/seed0
- completed paired cell 10/10: PS2-diagonal-minority-10/far/seed0

## Development outcome
- selected bounded accumulator: `kll-style-k128`
- ED2 ratio / historical online PSQT: `0.3191`
- ED2 ratio / selected paper: `0.0265`
- held-out SW1 ratio / historical online PSQT: `0.5263`
- rare-mode ED2 ratio / historical online PSQT: `0.0809`
- exact-pooled ED2 ratio / historical online PSQT: `0.2592`
- exact-pooled ED2 ratio / selected paper: `0.0215`
- family ED2 ratios, selected / historical:
  - `dependence`: `0.2054`
  - `gauss`: `0.3316`
  - `rare`: `0.0809`
  - `skew`: `0.4769`
- decision gates:
  - `finite_and_no_divergence`: **PASS**
  - `diagonal_better_than_historical`: **PASS**
  - `diagonal_better_than_paper`: **PASS**
  - `aggregate_ed2_better_than_historical`: **PASS**
  - `aggregate_sw1_better_than_historical`: **PASS**
  - `no_family_regression_over_5pct`: **PASS**
  - `median_excess_bridge_at_most_one_particle`: **PASS**
  - `rare_05_10_majority_recovered`: **PASS**
  - `bounded_memory_below_exact_pool`: **PASS**
- all development gates: **PASS**

## Interpretation boundary

The selected arm and every threshold were evaluated on reused development families. Exact pooling is an unbounded finite-stream ceiling. The KLL-style arm uses fixed-capacity random compactors and does not claim the optimal KLL space theorem. A fresh registry is required before a confirmatory superiority claim.
