# 03 — Laplacian formalization

**The headline result of the project.** Zero ℓ²-Laplace mean-shift drift
identifies *arbitrary* probability measures on *every* finite-dimensional
Euclidean space, machine-checked and axiom-free.

Entry theorem: `laplaceZeroDrift_identifies_euclidean` / `_rn` in
[LaplaceEuclideanConverse.lean](../../DriftingIdentifiability/LaplaceEuclideanConverse.lean).

## The G4 proof chain (the closed route)

The proof runs a global maximum principle for the companion-potential defect
`H = ψ_p − (Z_p/Z_q) ψ_q`.

- [LaplaceFoliationMaximum.lean](../../DriftingIdentifiability/LaplaceFoliationMaximum.lean) — the global maximum principle forcing `H ≡ 0`
- [LaplaceFoliationEndgame.lean](../../DriftingIdentifiability/LaplaceFoliationEndgame.lean) — gluing + total-mass endgame → `p = q`
- [LaplaceFoliationCancellation.lean](../../DriftingIdentifiability/LaplaceFoliationCancellation.lean), [LaplaceFoliationChart.lean](../../DriftingIdentifiability/LaplaceFoliationChart.lean), [LaplaceFoliationFactorization.lean](../../DriftingIdentifiability/LaplaceFoliationFactorization.lean), [LaplaceFoliationFlow.lean](../../DriftingIdentifiability/LaplaceFoliationFlow.lean) — the foliation machinery
- [LaplaceCompanion.lean](../../DriftingIdentifiability/LaplaceCompanion.lean), [LaplaceDisplacementHessian.lean](../../DriftingIdentifiability/LaplaceDisplacementHessian.lean), [LaplaceConeExtraction.lean](../../DriftingIdentifiability/LaplaceConeExtraction.lean), [LaplaceAtomAlignment.lean](../../DriftingIdentifiability/LaplaceAtomAlignment.lean), [LaplaceTiltedMeanMonotone.lean](../../DriftingIdentifiability/LaplaceTiltedMeanMonotone.lean)

## Radial program (earlier routes; now corollaries)

`LaplaceRadial*` — the n=2/n=3/general-n radial converses, slack removal, and
the ray/shell/system/invariance/measure infrastructure:
[Foundations](../../DriftingIdentifiability/LaplaceRadialFoundations.lean),
[FarField](../../DriftingIdentifiability/LaplaceRadialFarField.lean),
[Fourier](../../DriftingIdentifiability/LaplaceRadialFourier.lean),
ConverseN/2/3, DifferentiationN, IntegrabilityN, InvarianceN/2/3, MeasureN,
PhysicalBridgeN, RayN/2/3, ShellN/2/3, SlackAssociationN, SystemN/2/3,
ZonalBridgeN (all `LaplaceRadial*.lean` in `DriftingIdentifiability/`).

## Absolute-continuity / Abel route (alternative closure)

`LaplaceAC*` — [Abel](../../DriftingIdentifiability/LaplaceACAbel.lean),
[Asymptotics](../../DriftingIdentifiability/LaplaceACAsymptotics.lean),
[DensityRegularity](../../DriftingIdentifiability/LaplaceACDensityRegularity.lean),
[Propagation](../../DriftingIdentifiability/LaplaceACPropagation.lean),
[Regularity](../../DriftingIdentifiability/LaplaceACRegularity.lean),
[GaussianCertificate](../../DriftingIdentifiability/LaplaceACGaussianCertificate.lean),
[Final](../../DriftingIdentifiability/LaplaceACFinal.lean).

## General / atomic / unconditional converses and injectivity

[LaplaceGeneralConverse.lean](../../DriftingIdentifiability/LaplaceGeneralConverse.lean)
(+ Balance, CompanionWronskian, Endgame, NowhereDense, Reduction, Wronskian),
[LaplaceAtomicConverse.lean](../../DriftingIdentifiability/LaplaceAtomicConverse.lean),
[LaplaceAtomlessConverse.lean](../../DriftingIdentifiability/LaplaceAtomlessConverse.lean),
[LaplaceUnconditionalConverse.lean](../../DriftingIdentifiability/LaplaceUnconditionalConverse.lean),
[LaplaceRealConverse.lean](../../DriftingIdentifiability/LaplaceRealConverse.lean),
[LaplaceInjectivity.lean](../../DriftingIdentifiability/LaplaceInjectivity.lean),
[LaplaceEuclideanInjectivity.lean](../../DriftingIdentifiability/LaplaceEuclideanInjectivity.lean),
[LaplaceWronskian.lean](../../DriftingIdentifiability/LaplaceWronskian.lean),
[LaplacianGaussianConverse.lean](../../DriftingIdentifiability/LaplacianGaussianConverse.lean).

## Design notes & roadmaps (`.md`)

[LaplaceRnRoadmap](../../DriftingIdentifiability/LaplaceRnRoadmap.md),
[LaplaceG4Foliation](../../DriftingIdentifiability/LaplaceG4Foliation.md),
[LaplaceGeneralConverseRoadmap](../../DriftingIdentifiability/LaplaceGeneralConverseRoadmap.md),
[LaplaceArbitraryConverse](../../DriftingIdentifiability/LaplaceArbitraryConverse.md),
[LaplaceEndgame](../../DriftingIdentifiability/LaplaceEndgame.md),
[LaplaceHigherDim](../../DriftingIdentifiability/LaplaceHigherDim.md),
[LaplaceACDerivation](../../DriftingIdentifiability/LaplaceACDerivation.md),
[LaplaceACConditionalAxiomPlan](../../DriftingIdentifiability/LaplaceACConditionalAxiomPlan.md),
[LaplaceL5_HANDOFF](../../DriftingIdentifiability/LaplaceL5_HANDOFF.md),
[RawFieldConverse](../../DriftingIdentifiability/RawFieldConverse.md),
[PostConverseFrontier](../../DriftingIdentifiability/PostConverseFrontier.md)
(the resumable post-converse research spec).
