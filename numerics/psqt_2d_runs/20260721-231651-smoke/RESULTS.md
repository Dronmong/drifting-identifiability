# PSQT 2D development results

Status: exploratory development; not a frozen confirmation.

## Invariants
- PSQT mathematical/implementation invariants: PASS
- invariant paper: translation/finite PASS
- invariant snis: translation/finite PASS
- invariant paper matched-batch cancellation: PASS
- invariant paper-ND vs driftlab.compute_v_paper: PASS
- completed paired cell 1/8: PS2-GMM4-equal/concentrated/seed0
- completed paired cell 2/8: PS2-GMM4-equal/far/seed0
- completed paired cell 3/8: PS2-GMM5-unequal/concentrated/seed0
- completed paired cell 4/8: PS2-GMM5-unequal/far/seed0
- completed paired cell 5/8: PS2-GMM4-hetero/concentrated/seed0
- completed paired cell 6/8: PS2-GMM4-hetero/far/seed0
- completed paired cell 7/8: PS2-diagonal-dependence/concentrated/seed0
- completed paired cell 8/8: PS2-diagonal-dependence/far/seed0

## Development selection
- selected paper arm: `paper-tau2`
- selected PSQT arm: `psqt-L16-K32-R3-e0.5`
- target-balanced ED2 ratio, PSQT / selected paper: `0.1049`
- held-out SW1 ratio, PSQT / selected paper: `0.3142`
- ED2 ratio, PSQT / coordinate-PQT: `0.3517`
- ED2 ratio, PSQT / per-cell paper oracle: `0.1049`
- cell wins versus selected paper: `8/8`
- family ED2 ratios versus selected paper:
  - `dependence`: `0.2936`
  - `gauss`: `0.0744`

## Interpretation boundary

These outcomes are development evidence on reused target families. They select hyperparameters and cannot support a confirmatory claim. A fresh registry, frozen arm, target-level uncertainty, and full sample/work gates are required next.
