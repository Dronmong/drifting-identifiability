# Objective 7 numerics

Numerical evaluation of the machine-checked identifiability conditions at the
paper's actual operating point (arXiv:2602.04770v2, Table 8 and A.6).

Run (no install needed beyond [uv](https://docs.astral.sh/uv/)):

```
uv run --with numpy python numerics/run_all.py
```

Output: `numerics/RESULTS.md` (deterministic, seed 20260707).

## What this is and is not

This suite does **not** prove anything.  It evaluates the *proved* formulas at
concrete numbers, audits the formalization against the paper's own Algorithm-2
pseudo-code, and answers the Objective-7 question: which certified conditions
are realistic enough to matter, and where do the certified constants leave
practice behind.

The operating point is taken from the paper, not chosen for convenience:

- kernel `exp(-dist/tau_tilde)` with `tau_tilde = tau*sqrt(C)` on features
  normalized so that the mean pairwise distance is `sqrt(C)` (A.6, eqs. 18-22);
  in normalized units the kernel is `exp(-u/tau)` with typical `u ~ 1`;
- temperature grid `tau in {0.02, 0.05, 0.2}` (Table 8);
- per-class batch `N = Npos = Nneg = 64`, negatives = generated samples reused
  with the `eye(N)*1e6` self-mask (Algorithm 2);
- CFG scale `alpha in [1, 4]` (Table 8).

## Formula crosswalk (Python -> Lean)

| `driftlab.py` | Lean declaration |
|---|---|
| `gaussian_kernel` | `Paper.gaussianKernel` |
| `alg2_kernel` | `Algorithm2.algorithm2Kernel` (= `Paper.laplaceKernel`) |
| `interaction_matrix` | `inducedInteractionVector` at the empirical point basis (`basisInteraction_empirical2` / `EmpiricalFrameBound`) |
| `certified_frame_constant` | `gaussianEmpiricalPointCertifiedFrameConstant` |
| `frame_ceiling` | `gaussianEmpiricalPoint_frameConstant_le` |
| `frame_violation` | `InteractionFrameBound` (sup norm on the probe axis) |
| `u01_bare` | two-atom `U_01` via `basisInteraction_empirical2` (`empirical01Laplace` family) |
| `col_reweight_scale` | `inducedInteractionVector_columnReweighted01_eq_smul` |
| `column_reweighted_weight` | `algorithm2ColumnReweightedWeight` |
| `modified_field_two_atom` | limiting field of `ColumnReweightedMeanShiftFiniteSetup` |
| `bare_field_two_atom` | `Paper.meanShiftDrift` (eqs. 8/10) |
| `compute_v_paper` | paper Algorithm 2 pseudo-code (verbatim port) |
| `compute_v_lean` | `Paper.algorithm2Drift` (literal `finiteSoftmax` pipeline) |
| `centroid_diff` | `Algorithm2.noMaskCentroidDrift` / masked analogue |
| E3b bias | `deletedNegativeColumnWeight`, hypotheses of `deletedNegativeCentroid_meanSquare_le` |
| E3c delta | `maskPenaltyFactor`, `algorithm2Drift_sub_deletedDrift_norm_le` |
| E5 chain | `columnReweighted01_coefficientStability` + `estimate_failure_le_meanSquare` + `selfNormalizedIndexed_meanSquare_le` |
| E6 gate | `CFGTargetNonnegative` / `cfgTargetCoefficients` (`CFGAffine.lean`) |

## Experiments

- **E0** — transcription audit: two independent Algorithm-2 implementations
  agree to 1e-16; spot-checks of proved identities (matched-batch zero, the
  mass-product centroid form, the drift bound, frame-certificate validity).
- **E1** — certified frame constant vs basis size `m`, support window, and
  probe placement; the sqrt(2/e) separation sweet spot.
- **E2** — exact column-reweighting scale vs the proved `1/N` floor.
- **E3/E3b/E3c** — Monte-Carlo: `1/N` SNIS rate; the estimator's bias floor
  against the *bare* field (it targets the column-reweighted field); eye-mask
  leave-out bias `O(1/N)`; the self-mask perturbation scale.
- **E4** — softmax effective sample size per temperature at `N = 64` under a
  normalized-distance feature-geometry model: what each `tau` "sees".
- **E5** — the conditioning ledger: the full certified chain multiplied out
  into a sample-complexity number, against its LLN-typical and observed
  counterparts.
- **E6** — how often the CFG affine target is an actual probability vector at
  the paper's guidance scales.
