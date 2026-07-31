# Conditioned transport limitation-repair results

**Status:** implementation complete; proposed repair not promoted  
**Date:** 2026-07-23  
**Plan:** `ConditionedTransportLimitationRepairPlan.md`  
**Frozen protocol:** `ConditionedTransportLimitationRepairConfirmationProtocol.md`

## 1. Verdict

The investigation found and repaired a genuine approximation defect, but the
more accurate local field did **not** produce a statistically established
improvement over the current confirmed model.

The correct decision is:

> Retain the active-32, exact-atlas, fixed-level `M=128` conditioned-transport
> model as the primary. Do not promote the variance/adaptive representative
> system or the cross-fitted local controller.

The repaired representative system still beat the repository paper port very
strongly on every fresh target. Its failure is specifically a failure to
improve the already strong current model.

## 2. What was implemented

### 2.1 Reproducible limitation audit

`analyze_conditioned_transport_limitations.py` consumes the canonical frozen
primary and dense artifacts and emits:

- fixed-local hybrid/global ratios by dimension and family;
- compressed/dense ratios by dimension and family;
- representative field/mass/radius diagnostics by dimension;
- frozen-array support precision/recall diagnostics; and
- hashes of all input and output payloads.

The generated report is
`conditioned_transport_limitation_audit.json`.

### 2.2 Representative strategies

`projection_tree_representatives` now supports:

```text
fixed-level
variance-per-node
radius-priority
variance-with-tail-reserve
```

The historical algorithm remains the default control.

The new implementation records:

- strategy;
- split count;
- number of unique split directions;
- tail singletons;
- actual leaf-population range;
- partition projection products; and
- partition sorting work.

The equal-occupancy variance tree is vectorized by level. Every leaf chooses
the registered direction with greatest within-leaf projected variance.

### 2.3 Adaptive capacity

The runner can independently specify:

- a base representative count;
- a high-dimensional representative count; and
- the dimension where the larger capacity begins.

The selected candidate used `M=128` below dimension 8 and `M=256` at dimensions
8 and 16.

### 2.4 Cross-fitted local controller

A separate exact target atlas is constructed from deterministic controller
directions that are independent of the transport and final held-out banks.

At each macro-step, the controller:

1. reranks candidate teachers for local weights `{0, .05, .10, .25}`;
2. rejects candidates that exceed the global-only controller-tail tolerance;
3. identifies the best safe controller loss; and
4. selects the smallest safe near-optimal weight.

This is retained in the code as an ablation. It did not pass development.

### 2.5 Regression coverage

The tests now check:

- exact assignment and multiplicity;
- centers equal their leaf means;
- no empty leaves;
- deterministic replay;
- input-permutation stability away from ties;
- exact `M=N` field equivalence for every strategy;
- weighted self-mask semantics;
- anisotropic geometry detection by the variance tree;
- tail singleton creation;
- adversarial controller rejection; and
- aligned controller acceptance.

## 3. Reproduced initial diagnosis

The hashed post-hoc limitation audit reproduced two central patterns.

### 3.1 Fixed local weight

The local term helped every 8D and 16D target and all rare-GMM dimensions, but
hurt several 2D/4D common targets. Hybrid/global geometric ED2 ratios were:

| Dimension | Hybrid/global ED2 | Wins |
|---:|---:|---:|
| 2 | `0.9062` | `1/4` |
| 4 | `0.9247` | `2/4` |
| 8 | `0.6222` | `4/4` |
| 16 | `0.6937` | `4/4` |

The aggregate local gain is real, but a fixed weight is not universally
optimal.

### 3.2 Compression error grows with dimension

| Dimension | Median field relative-L2 error | Minimum cosine |
|---:|---:|---:|
| 2 | `0.0190` | `0.9997` |
| 4 | `0.0677` | `0.9921` |
| 8 | `0.2056` | `0.9438` |
| 16 | `0.3790` | `0.8915` |

The original tree uses only seven predetermined split directions for
`M=128`. The dimension trend therefore had a concrete implementation-level
explanation.

## 4. Consumed-registry representative development

Authoritative analysis:
`representative_strategy_development_analysis.json`.

### 4.1 Mechanism result

| Configuration | Geometric field error | Minimum cosine |
|---|---:|---:|
| fixed `M128` | `0.0878` | `0.9417` |
| variance `M128` | `0.0486` | `0.9589` |
| variance `M256` | `0.0249` | `0.9680` |
| radius `M128` | `0.0562` | `0.8656` |
| tail `M128` | `0.0581` | `0.8750` |

The variance tree was the only strategy that improved both aggregate error and
the lower-tail cosine. Radius and tail allocation reduced aggregate error but
created worse individual fields and were much slower.

### 4.2 Endpoint development result

Against fixed `M128`:

| Configuration | ED2 ratio | SW1 ratio |
|---|---:|---:|
| variance `M128` | `0.9652` | `0.9836` |
| variance `M256` | `0.9651` | `0.9819` |
| adaptive `M128/M256` | `0.9537` | `0.9754` |

These were consumed-registry results and were used only to select the fresh
candidate.

The vectorized variance implementation materially reduced its Python overhead.
Nevertheless, the larger adaptive capacity performs more kernel work than the
current `M128` model.

## 5. Cross-fitted controller result

Artifact:
`conditioned_transport_runs/20260723-143427-consumed/`  
Analysis:
`crossfit_controller_development_analysis.json`

The controller failed:

| Crossfit/fixed | Ratio | Wins |
|---|---:|---:|
| ED2 | `1.0457` | `6/16` |
| held-out SW1 | `1.0244` | `8/16` |

It remained better than global-only (`0.8412 / 0.9005`), so it did not remove
the local mechanism entirely. The problem was discrimination:

- mean selected local weight: `0.2341`;
- positive-weight macro-step rate: `0.9625`; and
- mean controller-reported one-step improvement: `7.54%`.

The controller believed the local term improved its independent one-step
criterion almost everywhere, but that criterion did not predict the later
amortized neural endpoint. The largest failures were nonlinear 4D/8D.

This falsifies the proposed controller, not the broader possibility of learned
or rollout-based control.

## 6. Fresh confirmation

Registry:
`conditioned_transport_repair_confirmation_registry.json`  
Registry SHA-256:
`f675c82a1f17aa991ba3c9f5b5f5555a4b57880269a5f504c64a7e96a91a1da4`

Artifacts:

- concentrated:
  `conditioned_transport_runs/20260723-144129-consumed/`;
- broad:
  `conditioned_transport_runs/20260723-144337-consumed/`.

Analysis:
`conditioned_transport_repair_confirmation_analysis.json`.

The independent unit was one of 32 new target instances. Concentrated and
broad initialization ratios were geometrically reduced within target before a
20,000-draw stratified bootstrap.

### 6.1 Candidate versus current model

| Metric | Geometric ratio | 95% interval | Wins |
|---|---:|---:|---:|
| ED2 | `0.9944` | `[0.9804, 1.0085]` | `20/32` |
| held-out SW1 | `0.9959` | `[0.9902, 1.0016]` | `20/32` |

The point estimates mildly favored the repair, but neither upper confidence
bound was below one. Gate A failed.

The effect was heterogeneous:

- all eight 16D target ratios favored the repair on ED2;
- all eight 16D target ratios favored the repair on SW1;
- the most serious regressions were the two 2D rare-GMM targets; and
- several 4D/8D cells were effectively tied.

This suggests that the geometric repair solves a real high-dimensional
problem but should not replace the fixed tree globally.

### 6.2 Mechanism gate

The mechanism gate passed:

- high-dimensional field-error ratio: `0.4911`;
- candidate minimum cosine: `0.9485`;
- control minimum cosine: `0.7753`;
- median row-mass error: `0.0612` versus `0.1618`; and
- median column-mass error: `0.1311` versus `0.2213`.

The approximation repair worked mathematically and numerically. The failure is
the implication “more accurate local field produces a better amortized
generator.”

### 6.3 Support gate

Median support diagnostics were essentially tied:

| Metric | Repaired | Current |
|---|---:|---:|
| precision | `0.9631` | `0.9629` |
| recall | `0.9814` | `0.9814` |
| normalized mean error | `0.0482` | `0.0497` |
| covariance error | `0.1822` | `0.1720` |

However:

- repaired minimum mode coverage: `0.5`;
- current minimum mode coverage: `1.0`;
- repaired median rare-mass error: `0.00728`; and
- current median rare-mass error: `0.00610`.

The rare-mass tolerance passed, but minimum coverage failed. Gate D failed.

### 6.4 Retained advantage over paper

The repair remained decisively better than paper:

| Metric | Repaired/paper ratio | 95% interval | Wins |
|---|---:|---:|---:|
| ED2 | `0.2709` | `[0.2500, 0.2936]` | `32/32` |
| held-out SW1 | `0.5033` | `[0.4810, 0.5266]` | `32/32` |

Gate C passed. This is a robustness replication of the conditioned-transport
advantage, but not an improvement over the current candidate.

### 6.5 Cost

Relative to current fixed `M128`, the adaptive candidate had median ratios:

| Cost | Ratio |
|---|---:|
| kernel pairs | `1.50` |
| projection products | `1.287` |
| online CPU wall | `1.527` |
| setup + training CPU wall | `1.371` |

The diagnostic calls remain inside wall time, but the increased pair and
projection counts are genuine. The new system is not a cost improvement.

## 7. Scientific interpretation

### 7.1 Compression is acting as regularization

The historical tree is not merely a cheap approximation to the dense local
field. Its bias changes the optimization trajectory and can be beneficial.

The fresh result shows:

```text
better local-field approximation
    does not imply
better final neural distribution
```

This is an important correction to the original repair hypothesis.

### 7.2 The limitation is dimension- and family-dependent

The variance/adaptive tree was consistently helpful in 16D but harmful on
2D rare mixtures. A universal replacement is therefore unjustified.

A future development study may test a label-free gate that retains the fixed
tree when its radius/error proxy is already small and activates the variance
tree only when approximation difficulty is high. The current fresh registry
is now consumed and may not confirm that new rule.

### 7.3 One-step validation is not enough

The cross-fitted controller's independent quantile score did not predict the
outcome after repeated teacher construction and neural amortization. A future
controller would need a short rollout, a learned endpoint predictor trained on
separate tasks, or a stronger causal trust-region certificate.

## 8. Final project state

Completed:

- reproducible limitation audit;
- four representative strategies;
- adaptive capacity;
- exact weighted/self-mask integration;
- real cost ledgers;
- expanded invariant suite;
- cross-fitted controller ablation;
- consumed development comparison;
- frozen 32-target/two-initialization confirmation; and
- hashed machine-readable analyses.

Not promoted:

- variance/adaptive representative replacement;
- radius-priority representatives;
- tail-reserve representatives; and
- cross-fitted local controller.

The strongest current model and claim remain those in
`ProjectionKernelOptimizationConfirmationResults.md`.
