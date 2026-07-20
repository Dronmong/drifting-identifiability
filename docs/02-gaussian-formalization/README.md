# 02 — Gaussian formalization

Converses and supporting analysis for the **Gaussian** kernel, plus the
convolution-injectivity machinery reused by the jittered NCJ theorem (T2).

## Core files

- [GaussianArbitraryConverse.lean](../../DriftingIdentifiability/GaussianArbitraryConverse.lean) — zero Gaussian mean-shift drift identifies arbitrary measures (all-dimension Gaussian converse)
- [GaussianConvolutionInjectivity.lean](../../DriftingIdentifiability/GaussianConvolutionInjectivity.lean) — convolution by a scaled standard Gaussian is injective on finite measures (`Measure.eq_of_conv_scaledStdGaussian_eq`); the characteristic-function cancellation used by NCJ T2
- [GaussianScoreRecovery.lean](../../DriftingIdentifiability/GaussianScoreRecovery.lean) — score-function recovery for the Gaussian case
- [GaussianNondegeneracy.lean](../../DriftingIdentifiability/GaussianNondegeneracy.lean) — nondegeneracy conditions for the Gaussian setup
- [LaplacianGaussianConverse.lean](../../DriftingIdentifiability/LaplacianGaussianConverse.lean) — the Laplace-kernel → Gaussian bridge (also indexed under Laplacian); design note [LaplacianGaussianConverse.md](../../DriftingIdentifiability/LaplacianGaussianConverse.md)

## Relationship to the other converses

The Gaussian converse was closed earlier and is now one of several fully
instantiated kernels; the general ℝⁿ result lives under
[03 — Laplacian formalization](../03-laplacian-formalization/README.md). The
Gaussian-convolution injectivity here is what lets the **jittered** NCJ field
(T2, in [04](../04-empirical-experimentation/README.md)) retain an end-to-end
identifiability guarantee.
