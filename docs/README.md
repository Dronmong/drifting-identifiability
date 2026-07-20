# Project map

This `docs/` tree is the **navigational index** for the project. It cleanly
divides every result into categories and links to the real source files.

> **Why an index and not physically-moved files.** The Lean source is a single
> certified library: in Lean 4 a file's folder *is* its module name, so moving a
> file rewrites every `import` and risks the trust audit (`scripts/Check.ps1`,
> which must stay green). The Python experiments are frozen: each run directory
> verifies its source files by path + SHA-256, so moving them breaks
> reproducibility. The certified source therefore stays in place; this hub gives
> the clean division by reference. Physical reorganization, if ever wanted, must
> be a separate build-verified operation.

## Categories

| # | Category | What lives here |
|---|---|---|
| 01 | [Appendix-heuristic formalization](01-appendix-heuristic-formalization/README.md) | Appendix C.1 finite identifiability, the Algorithm-2 estimator, frame bounds, self-normalized (SNIS) consistency |
| 02 | [Gaussian formalization](02-gaussian-formalization/README.md) | Gaussian-kernel converses, convolution injectivity, score recovery |
| 03 | [Laplacian formalization](03-laplacian-formalization/README.md) | **The headline result:** the full ℝⁿ ℓ²-Laplace converse `V=0 ⟹ p=q`, and its whole proof chain |
| 04 | [Empirical experimentation](04-empirical-experimentation/README.md) | All numerics: collapse atlas, Phase C, low-dim studies, the NCJ program, and the **[ideas ledger](04-empirical-experimentation/PaperImprovementAttempts.md)** |
| 05 | [Misc / infrastructure](05-misc/README.md) | Trust boundary, axioms, failure cases, scripts, top-level status/presentation |

## The one-paragraph state of the project

The **core achievement is the Laplacian formalization (03)**: zero ℓ²-Laplace
mean-shift drift identifies *arbitrary* probability measures in *every* finite
dimension, machine-checked and axiom-free (`laplaceZeroDrift_identifies_rn`).
The **empirical program (04)** asked whether the identifiability theory yields a
*better algorithm* than the paper; after a disciplined, pre-registered sequence
it did **not** produce a deployment-relevant improvement (see the ideas ledger
for every attempt and the pattern). Project-wide status:
[ResearchStatus.md](../DriftingIdentifiability/ResearchStatus.md).
