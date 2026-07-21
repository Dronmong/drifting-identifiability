# Objective 7 numerics

Numerical evaluation of the machine-checked identifiability conditions at the
paper's actual operating point (arXiv:2602.04770v2, Table 8 and A.6).

Run (no install needed beyond [uv](https://docs.astral.sh/uv/)):

```
uv run --with numpy python numerics/run_all.py
```

Output: `numerics/RESULTS.md` (deterministic, seed 20260707).

Real encoder features:

```
uv run --with numpy python numerics/real_feature_diagnostics.py \
  --features path/to/features.npy \
  --m 8 --num-probes 64 --taus 0.02 0.05 0.2
```

The feature file should be a `.npy` array of shape
`[num_samples, feature_dim]`, or a `.npz` archive containing such an array
(select with `--key`; otherwise the first array is used). The runner centers
and rescales features by default so the sampled mean pairwise distance is one,
matching the normalized units used elsewhere in this numerical ledger. It
reports softmax effective sample sizes, bare interaction-matrix conditioning,
and column-reweighted interaction-matrix conditioning. Large feature tensors
and generated `REAL_FEATURES*.md` reports are ignored by Git by default; commit
only small sanitized summaries intentionally.

## What this is and is not

This suite does **not** prove anything.  It evaluates the *proved* formulas at
concrete numbers, audits the formalization against the paper's own Algorithm-2
pseudo-code, and answers the Objective-7 question: which certified conditions
are realistic enough to matter, and where do the certified constants leave
practice behind.

## Dynamics Phase C validation

The audit-corrected finite-particle dynamics benchmark is separate from the
earlier identifiability ledger:

* `PhaseCValidation.md` specifies the controlled C1--C3 experiments;
* `driftbench_v2.py` is the exact executable harness;
* `DesignRules.md` records the qualified conclusions;
* `bench_runs_v2/` contains standard-profile manifests, source snapshots,
  per-seed data, and trajectories;
* `bench_figs_v2/` contains the corresponding uncertainty plots.

The old `PhaseC_DesignRules.md`, `driftbench.py`, and `bench_runs/` are retained
as an explicitly superseded first pass. The V2 results are synthetic particle
experiments; they do not claim trained-model or real-encoder performance.

## Low-dimensional base-versus-modified study

The next-stage matched Algorithm-2 study is documented in:

* `LowDimPerformanceRoadmap.md` and `LowDimPerformanceResults.md` for the
  historical D0--D3 negative result;
* `LowDimAttributionProtocol.md` and `lowdim_attribution.py` for the
  audit-corrected fresh bandwidth × mask × step experiment;
* `LowDimAttributionResults.md` for the fresh held-out result.

The fresh global gate also failed, so no aggregate outperformance or learned-
generator claim is made. The surviving empirical finding is conditional: with
bandwidth and step fixed, disabling the eye mask substantially improves ED² on
fresh curved ring/circle/moon targets, while a cluster-count trigger can
misfire on a finite Gaussian mixture.

The operating point is taken from the paper, not chosen for convenience:

- kernel `exp(-dist/tau_tilde)` with `tau_tilde = tau*sqrt(C)` on features
  normalized so that the mean pairwise distance is `sqrt(C)` (A.6, eqs. 18-22);
  in normalized units the kernel is `exp(-u/tau)` with typical `u ~ 1`;
- temperature grid `tau in {0.02, 0.05, 0.2}` (Table 8);
- per-class batch `N = Npos = Nneg = 64`, negatives = generated samples reused
  with the `eye(N)*1e6` self-mask (Algorithm 2);
- CFG scale `alpha in [1, 4]` (Table 8).

## Quantile-to-Laplace learned-generator program

The later one-dimensional learned-generator program is recorded separately:

* `QLDNextGenerationResearch.md` develops the large-batch successor to the
  original quantile-to-Laplace candidate;
* `LBQCDDevelopmentResults.md` records the development selection;
* `LBQCDConfirmatoryProtocol.md` and `LBQCDConfirmatoryResults.md` contain the
  frozen confirmatory protocol and its scoped 17.8% ED2 improvement;
* `OccupancyAdaptiveQuantileResearch.md` audits the hypothesis that
  large-batch transport could improve coarse mass allocation more reliably
  than final error and specifies the state-aware, stratified successor;
* `OASQDDevelopmentProtocol.md`, `oasqd.py`, and
  `run_oasqd_development.py` implement that fresh staged campaign;
* `OASQDDevelopmentResults.md` records the negative O5 decision: the atlas and
  unbiased stratified estimator worked, but the frozen OA-SQD candidate tied
  QLD rather than surpassing it, so no confirmation was launched.

The earlier cross-arm LB-QCD coverage-time comparison used unequal event
sample sizes and is withdrawn. The OA-SQD runner uses the same independent
event-probe size for every arm; endpoint LB-QCD results are unaffected by this
correction.

Neither frozen LB-QCD registry may be reused to tune the successor. The result
is limited to the documented one-dimensional synthetic generator benchmark
and does not assert ImageNet, real-feature, or multidimensional superiority.

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
| E5 chain | `columnReweighted01_coefficientStability` + `estimate_failure_le_meanSquare` + `selfNormalizedIndexed_meanSquare_le` + `selfNormalizedIndexed_deviation_prob_le` |
| E6 gate | `CFGTargetNonnegative` / `cfgTargetCoefficients` (`CFGAffine.lean`) |
| `real_feature_diagnostics.py` | empirical finite matrix for `InteractionFrameBound`; column-reweighted matrix for `ColumnReweightedMeanShiftFiniteSetup` |

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
  into a sample-complexity number, comparing the old deterministic denominator
  floor, the new high-probability denominator-tail refinement, the LLN-typical
  benchmark, and observed Monte-Carlo scaling.
- **E6** — how often the CFG affine target is an actual probability vector at
  the paper's guidance scales.
- **Real features** — `real_feature_diagnostics.py` runs the same frame and
  Algorithm-2 column-reweighting diagnostics on externally supplied encoder
  features.
