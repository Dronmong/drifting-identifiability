# 01 — Appendix-heuristic formalization

The paper's **Appendix C.1** heuristic chain made rigorous: finite-coefficient
identifiability, the interaction-vector expansion, the Algorithm-2 estimator,
frame/stability bounds, and self-normalized (SNIS) consistency. This is the
finite-dimensional, estimator-level backbone the empirical program builds on.

## Finite identifiability & interaction frame (Appendix C.1 core)

- [PaperFiniteIdentifiability.lean](../../DriftingIdentifiability/PaperFiniteIdentifiability.lean) — Eqs (29)–(31): finite coefficients, induced interaction vectors `Uᵢⱼ`, antisymmetry, the expansion `∑ aᵢbⱼUᵢⱼ`
- [FiniteStability.lean](../../DriftingIdentifiability/FiniteStability.lean) — `InteractionFrameBound`, strict-pair scaling, quantitative coefficient stability (used by NCJ T1)
- [EmpiricalFrameBound.lean](../../DriftingIdentifiability/EmpiricalFrameBound.lean), [FiniteGrouping.lean](../../DriftingIdentifiability/FiniteGrouping.lean), [FiniteLegitimacy.lean](../../DriftingIdentifiability/FiniteLegitimacy.lean)
- [PopulationIdentifiability.lean](../../DriftingIdentifiability/PopulationIdentifiability.lean), [PopulationCandidate.lean](../../DriftingIdentifiability/PopulationCandidate.lean), [CandidateConditions.lean](../../DriftingIdentifiability/CandidateConditions.lean), [PaperDriftIdentifiability.lean](../../DriftingIdentifiability/PaperDriftIdentifiability.lean), [CharacteristicIdentifiability.lean](../../DriftingIdentifiability/CharacteristicIdentifiability.lean)

## Algorithm-2 estimator & self-normalized consistency (Objective 4)

- [Algorithm2Estimator.lean](../../DriftingIdentifiability/Algorithm2Estimator.lean) — the bi-softmax `compute_V`; the audited factorization `V = (P·Q)·(Cpos−Cneg)` (basis of NCJ T1)
- [Algorithm2SNIS.lean](../../DriftingIdentifiability/Algorithm2SNIS.lean) — Algorithm 2 as a self-normalized importance-sampling estimator; centroid mean-square bounds (basis of NCJ T3)
- [SelfNormalizedConsistency.lean](../../DriftingIdentifiability/SelfNormalizedConsistency.lean) — abstract SNIS L²-consistency
- [ColumnReweightedMeanShift.lean](../../DriftingIdentifiability/ColumnReweightedMeanShift.lean), [ColumnReweightedTwoAtom.lean](../../DriftingIdentifiability/ColumnReweightedTwoAtom.lean) — the column-reweighted kernel and two-atom instantiation
- [SelfMaskPerturbation.lean](../../DriftingIdentifiability/SelfMaskPerturbation.lean), [DeletedEstimatorConsistency.lean](../../DriftingIdentifiability/DeletedEstimatorConsistency.lean), [DenominatorTail.lean](../../DriftingIdentifiability/DenominatorTail.lean) — eye-mask perturbation, deleted (leave-masked-out) estimator, denominator tail control
- [FiniteSampleBridge.lean](../../DriftingIdentifiability/FiniteSampleBridge.lean) — estimator-agnostic finite-sample → identifiability bridge

## Model-class infrastructure (Objective 3)

- [FeatureSpaceIdentifiability.lean](../../DriftingIdentifiability/FeatureSpaceIdentifiability.lean), [PracticalModelClasses.lean](../../DriftingIdentifiability/PracticalModelClasses.lean), [SmoothBumpBasis.lean](../../DriftingIdentifiability/SmoothBumpBasis.lean), [BalancedSampling.lean](../../DriftingIdentifiability/BalancedSampling.lean), [SinkhornBalanced.lean](../../DriftingIdentifiability/SinkhornBalanced.lean), [Conditional.lean](../../DriftingIdentifiability/Conditional.lean), [CFGAffine.lean](../../DriftingIdentifiability/CFGAffine.lean), [Extensions.lean](../../DriftingIdentifiability/Extensions.lean)

## Design note

[WrittenProof.md](../../DriftingIdentifiability/WrittenProof.md) — the informal
proof this layer formalizes.
