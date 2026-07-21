# 04 — Empirical testing and experimentation

All numerics: the research culture of "probe the drifting model, then try to
beat the paper under pre-registered gates." The strategic centerpiece here is
the **[ideas ledger](PaperImprovementAttempts.md)** — every modification we
tried against the paper metrics, its outcome, and the pattern across them.

## ⭐ Start here

- **[PaperImprovementAttempts.md](PaperImprovementAttempts.md)** — the ledger of every idea tried to beat the paper, with the cross-program pattern analysis and the resulting next target.
- **[QuantileFissionConfirmatoryResults.md](../../numerics/QuantileFissionConfirmatoryResults.md)** — completed frozen assessment of 1-D quantile-to-Laplace drifting: a modest ED2/SW1 and compute improvement, but a failed pre-registered minimum-effect gate.
- **[QLDNextGenerationResearch.md](../../numerics/QLDNextGenerationResearch.md)** — forensic analysis of QLD's confirmed strengths and weaknesses, literature cross-check, and the recommended large-batch/quantile-calibrated successor.
- **[LBQCDDevelopmentResults.md](../../numerics/LBQCDDevelopmentResults.md)** — implemented successor: target-gated virtual-large-batch quantile transport reaches `.7948` ED2 versus the selected paper baseline and `.8685` versus its hindsight oracle on a new development suite, with explicit compute and far-start limitations.
- **[QuantileFissionDriftingPlan.md](../../numerics/QuantileFissionDriftingPlan.md)** — candidate design, exploratory evidence, and the now-completed frozen protocol.
- **[ModeRecoveryRoadmap.md](../../numerics/ModeRecoveryRoadmap.md)** — the closed characterization that isolated the fission barrier motivating the active candidate.

## Certified dynamics layer (Lean, but empirically motivated)

- [NCJIdentifiability.lean](../../DriftingIdentifiability/NCJIdentifiability.lean) — T1–T4 for the NCJ program (positive-gain zero-set, jittered converse, cross-fit consistency, no-freeze vs attenuation)
- [LaplaceFissionInstability.lean](../../DriftingIdentifiability/LaplaceFissionInstability.lean) — Phase A: fission instability of point collapse
- [FiniteRangeMassBlindness.lean](../../DriftingIdentifiability/FiniteRangeMassBlindness.lean) — Phase A: finite-range kernels are mass-blind

## Programs (chronological)

### Collapse Atlas + Phase C (dynamics design rules)
- Code: `numerics/collapse_atlas.py`, `collapse_atlas2.py`, `collapse_atlas3.py`, `atlas_e3b_check.py`, `atlas_p2b_inspect.py`, `driftbench.py`, `driftbench_v2.py`, `frontier_screen.py`, `phase_c_v2_figs.py`, `bench_figs_gen.py`
- Docs: [CollapseAtlas](../../numerics/CollapseAtlas.md), [CollapseAtlasAudit](../../numerics/CollapseAtlasAudit.md), [CollapseAtlasResults](../../numerics/CollapseAtlasResults.md), [PhaseCValidation](../../numerics/PhaseCValidation.md), [DesignRules](../../numerics/DesignRules.md), [PhaseC_DesignRules](../../numerics/PhaseC_DesignRules.md)

### Low-dimensional performance study (D0–D3) + fresh attribution
- Code: `numerics/lowdim_drift.py`, `lowdim_benchmark.py`, `lowdim_generator.py`, `lowdim_attribution.py`
- Docs: [LowDimPerformanceRoadmap](../../numerics/LowDimPerformanceRoadmap.md), [LowDimPerformanceProtocol](../../numerics/LowDimPerformanceProtocol.md), [LowDimPerformanceResults](../../numerics/LowDimPerformanceResults.md), [LowDimAttributionProtocol](../../numerics/LowDimAttributionProtocol.md), [LowDimAttributionResults](../../numerics/LowDimAttributionResults.md)

### NCJ program (normalized, cross-fitted, jittered drifting)
- Code: `numerics/identifiability_drift.py` (field library + invariant tests), `run_identifiability_improvement.py` (particle gate E4), `run_identifiability_generator.py` (generator gate E5), `ncj_abc_comparison.py`, `generator_optimizer_diagnostic.py`, `generator_tempered_gain_screen.py`
- Docs: [IdentifiabilityDrivenImprovementPlan](../../numerics/IdentifiabilityDrivenImprovementPlan.md), [IdentifiabilityImprovementProtocol](../../numerics/IdentifiabilityImprovementProtocol.md), [IdentifiabilityImprovementResults](../../numerics/IdentifiabilityImprovementResults.md), [IdentifiabilityGeneratorProtocol](../../numerics/IdentifiabilityGeneratorProtocol.md), [GeneratorTransferResearch](../../numerics/GeneratorTransferResearch.md)

### Foundational / kernel screening
- Code: `numerics/driftlab.py`, `rn_g2.py`, `rn_screen.py`, `rn_screen2.py`, `rn_screen3.py`, `rn_screen4.py`, `real_feature_diagnostics.py`, `run_all.py`
- Docs: [numerics/README](../../numerics/README.md), [numerics/RESULTS](../../numerics/RESULTS.md)

## Run artifacts

Frozen run directories (manifests, rows, source snapshots, hashes) live under
`numerics/lowdim_runs/`, `numerics/atlas_runs/`, and
`numerics/identifiability_runs/`. They are immutable audit trail; do not edit.
