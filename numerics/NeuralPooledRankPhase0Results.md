# Neural pooled-rank Phase-0 checkpoint

**Status:** engineering and mathematical regression layer complete  
**Date:** 2026-07-22  
**Research plan:** `KLLPSQTNeuralAmortizationResearch.md`

## What was implemented

`neural_pooled_rank.py` now provides the reusable foundation for neural
amortization of persistent pooled-rank supervision:

- an immutable `QuantileAtlas` with validation and midpoint interpolation;
- exact inverted-ECDF target-atlas construction;
- Apache DataSketches 5.2.0 KLL target-atlas construction, including serialized
  per-direction sketch states and the reported rank-error scale;
- detached global projected-rank assignment with stable tie handling;
- the scaled empirical sliced-2-Wasserstein neural loss;
- the free-particle PSQT correction under the same convention;
- frame-operator rank, condition, and tightness diagnostics;
- Torch CPU/CUDA RNG capture and replay; and
- two-pass Run-Sort-ReRun microbatch backpropagation that accumulates gradients
  but cannot prematurely perform an optimizer step.

The target atlas is allowed to persist. Generated rank assignments are not:
the API recomputes them from a current effective generated population for each
RSR call.

`neural_pooled_rank_tests.py` is a standalone deterministic Phase-0 regression
suite. It does not consume any development or confirmatory target registry.

## Validation command

The Windows-compatible pinned command is:

```text
uv run --python 3.12 --with torch==2.7.1 --with numpy \
  --with datasketches==5.2.0 \
  python numerics/neural_pooled_rank_tests.py
```

The recorded environment was:

```text
Python              3.12
Torch               2.7.1+cpu
NumPy               2.5.1
Apache DataSketches 5.2.0
CUDA                 unavailable on this machine
```

Ruff linting and formatting checks also pass for both new Python files.

## Regression results

All seven test groups passed:

| Check | Observed result |
|---|---:|
| Exact midpoint atlas vs sorted projections | max error `0.0` |
| No-compaction Apache KLL vs exact inverted ECDF | max error `0.0` |
| Autograd vs central finite differences | max error `8.998e-11` |
| PSQT correction vs `-B * grad J_B` | max error `3.331e-16` |
| Existing PSQT one-step reconstruction vs new correction | max error `2.220e-16` |
| Full-activation vs microbatch RSR loss | error `2.220e-16` |
| Full-activation vs microbatch RSR parameter gradients | max error `5.551e-17` |
| RSR gradients, one chunk vs five chunks | max error `5.551e-17` |
| Full vs chunked deterministic Run features | max error `5.551e-17` |
| Dropout Run/ReRun with RNG replay | finite nonzero gradients, PASS |
| Atom, support-gap, 1% rare, and 5% rare cases | finite and deterministic, PASS |
| Coordinate frame | rank `3`, condition `1.0` |
| Deliberately duplicated one-axis frame | rank `1`, infinite condition |

The KLL wrapper reported normalized rank error `0.02051784`, consistent with
the earlier independent accumulator audit. The edge-case tests do not claim
that KLL accurately resolves a 1% rare component; they verify only finite,
deterministic behavior and observed projected support. Rank accuracy still does
not imply value accuracy across a support gap.

## What this establishes

The implementation now has an internally consistent mathematical convention:

```text
J_B = d / (2 L B) * sum squared matched projection residuals
PSQT correction = -B * feature-gradient of J_B
```

RSR recovers the full-population neural gradient while keeping activation
memory at the microbatch scale, provided that replayed outputs are unchanged.
The strict replay check catches unhandled stochastic or stateful behavior;
Torch dropout is supported through per-chunk RNG replay. NumPy/Python random
state and training-mode stateful buffers such as BatchNorm are intentionally
not silently managed.

## What this does not establish

This checkpoint contains no neural performance experiment. It does not show:

- that a shared generator can match the free-particle PSQT result;
- improvement over the paper's model in any output metric;
- scaling beyond the tested algebraic tensor dimensions;
- GPU replay equivalence;
- image or frozen-real-feature performance;
- finite-direction identifiability; or
- reliable KLL recovery below its rank-resolution scale.

## Next implementation checkpoint

The next step is Phase 1 from the research plan: generate a new, unconsumed
synthetic registry in dimensions `2, 4, 8, 16` and compare, under matched
budgets:

1. a neural paper-field baseline;
2. ordinary small-minibatch sliced-Wasserstein training;
3. exact-atlas RSR;
4. KLL-atlas RSR;
5. a small-effective-population KLL ablation;
6. a registered paper-plus-pooled-rank hybrid; and
7. a free-particle exact-atlas ceiling measuring the amortization gap.

The protocol and registry must be frozen before running performance comparisons.
