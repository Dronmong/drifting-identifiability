# Dimension-Adaptive Rollout Confirmation Results

**Status:** primary gates passed; candidate promoted for the scoped synthetic
quality result  
**Date:** 2026-07-23  
**Frozen protocol:** `AdaptiveRolloutConfirmationProtocol.md`  
**Machine-readable analysis:** `adaptive_rollout_confirmation_analysis.json`

Protocol SHA-256:
`909c04f81af447ff0aa5f73256a2e27645fffb77c69c573db0ea87570a84107a`.

## 1. Verdict

The repair diagnosis was correct: the principal remaining high-dimensional
defect was the distance traveled by the same free particles before neural
amortization, not another representative-tree approximation error.

The selected dimension-adaptive rollout improved the prior strongest control
on both primary metrics on an unseen 32-target registry:

| Candidate / comparator | ED2 ratio (95% bootstrap CI) | SW1 ratio (95% bootstrap CI) |
|---|---:|---:|
| prior fixed control | **0.8570** (`0.8097`, `0.9071`) | **0.9243** (`0.9017`, `0.9477`) |
| previous gated hybrid | **0.8831** (`0.8384`, `0.9332`) | **0.9359** (`0.9138`, `0.9583`) |
| matched paper implementation | **0.2689** (`0.2469`, `0.2931`) | **0.4967** (`0.4741`, `0.5199`) |

Lower is better. The confidence intervals are target-level, stratified over
dimension and family, and reduce concentrated/broad initializations within
each target. All preregistered primary upper bounds are below one.

The correct scoped claim is:

> On this fresh 2D/4D/8D/16D synthetic neural-generator benchmark, persistent
> same-particle reranking only where dimensional sparsity warrants it improves
> the previous implementation and the matched paper comparator in ED2 and
> held-out sliced Wasserstein quality.

It is not a compute win or an image-scale result.

## 2. What was changed

### 2.1 Persistent free-particle teacher

`reranked_psqt_rollout` repeatedly:

1. evaluates the current projected ranks of the same 512 free particles;
2. assigns target quantiles from the frozen target atlas;
3. takes a feature-space transport substep; and
4. reranks before the next substep.

No generator evaluation occurs inside the rollout. Only the final teacher is
amortized, so generator-example count and generator forward-call count remain
matched.

### 2.2 Frozen dimension-adaptive horizon

The consumed development screen showed a sharp dimension split. The final
candidate therefore uses:

| Dimension | Rollout substeps |
|---:|---:|
| 2 | 1 |
| 4 | 1 |
| 8 | 2 |
| 16 | 4 |

Every substep has size `0.5`. The rule depends only on dimension. It cannot
inspect target labels, candidate outputs, or evaluation metrics.

At 2D and 4D, the candidate is exactly the fixed control. Across both fresh
initializations, the maximum difference in ED2, held-out SW1, and training
quantile RMSE was exactly zero in all 32 low-dimensional rows.

### 2.3 Honest rare-core diagnostics

The runner now calibrates a 95% component core using the independent target
reference and separately records:

- output mass inside the rare core;
- nearest-rare mass outside the core (bridge mass);
- binary component-core coverage;
- rare nearest-component mass error;
- teacher rare-core hits; and
- generator rare-core retention.

These diagnostics are not used for training or routing.

### 2.4 Tested but unpromoted safeguards

The implementation also supports:

- per-particle local-field safety against the frozen projected residual; and
- displacement-stratified student microbatch ordering.

The consumed screen rejected their combined universal use. The safety rule
reduced the average local weight to about `0.074`, increased projection work
substantially, and did not deliver consistent endpoint quality. These remain
audited ablations, not part of the promoted candidate.

## 3. Development decision

The all-dimension rollout arms exposed the failure boundary:

| Consumed candidate / fixed | ED2 ratio | SW1 ratio |
|---|---:|---:|
| two-step everywhere | `1.0918` | `1.0303` |
| four-step everywhere | `1.2276` | `1.0789` |
| four-step + safety + balance | `1.0441` | `1.0404` |

Those aggregates fail because repeated transport overmoves several 2D/4D
targets. In contrast, the same two-step rollout at 8D had ED2/SW1 ratios
`0.7697 / 0.8847`, and the four-step rollout at 16D had
`0.7218 / 0.8273`.

Freezing one/two/four steps by dimension before confirmation preserved the
low-dimensional result and retained the high-dimensional gain. On the
consumed reconstruction its aggregate ratios were `0.8633 / 0.9249`.

## 4. Fresh confirmation design

Registry:
`adaptive_rollout_confirmation_registry.json`  
Registry SHA-256:
`caeac33206123783cecb1ccc2954ab648056e6508a56fe60eb5025718b33f4bf`

The registry contains two new instances in every dimension/family cell:

- dimensions: 2, 4, 8, 16;
- families: balanced GMM, rare GMM, correlated Student-t, nonlinear;
- 32 independent targets;
- concentrated and broad initializations;
- 64 registered training directions and 128 held-out directions.

Artifacts:

- concentrated:
  `conditioned_transport_runs/20260723-161850-consumed/`;
- broad:
  `conditioned_transport_runs/20260723-162201-consumed/`.

Both artifacts passed deep recomputation of all 128 outputs, registry/source
hash checks, complete-cell checks, and the generator/kernel/projection/sort
ledgers.

## 5. Quality results

### 5.1 Candidate versus prior control by dimension

| Dimension | ED2 ratio | SW1 ratio |
|---:|---:|---:|
| 2 | `1.0000` | `1.0000` |
| 4 | `1.0000` | `1.0000` |
| 8 | **`0.6853`** | **`0.8287`** |
| 16 | **`0.7873`** | **`0.8809`** |

The candidate won ED2 on 14 of the 16 affected high-dimensional targets and
SW1 on 13 of 16. The other 16 targets are exact low-dimensional ties. Thus the
aggregate improvement does not come from sacrificing low-dimensional quality.

### 5.2 Candidate versus prior control by family

| Family | ED2 ratio | SW1 ratio |
|---|---:|---:|
| balanced GMM | **`0.7961`** | **`0.8890`** |
| correlated Student-t | **`0.9057`** | **`0.9395`** |
| nonlinear | **`0.8144`** | **`0.9153`** |
| rare GMM | **`0.9188`** | **`0.9549`** |

Every registered family improves on both metrics after equal weighting over
its dimensions and targets. This is materially stronger than a single-family
or aggregate-only gain.

### 5.3 Candidate versus paper comparator

The candidate beat the matched paper implementation on all 32 targets for
both ED2 and SW1. Per-dimension ED2 ratios were:

| Dimension | ED2 ratio | SW1 ratio |
|---:|---:|---:|
| 2 | `0.2177` | `0.4416` |
| 4 | `0.2625` | `0.4919` |
| 8 | `0.2830` | `0.5263` |
| 16 | `0.3234` | `0.5321` |

This comparison uses the repository's matched optimized paper-neural arm at
the same generator-example budget. It is not a reproduction of the paper's
ImageNet evaluation.

### 5.4 Training criterion

Training quantile RMSE also improved:

- candidate/fixed: `0.9415` (95% CI `0.9211`–`0.9623`);
- candidate/gated: `0.9482` (95% CI `0.9284`–`0.9684`);
- candidate/paper: `0.4282` (95% CI `0.4066`–`0.4509`).

Agreement of training and held-out endpoints reduces concern that the result
is merely overfitting the registered projection bank.

## 6. Rare-component mechanism

The new diagnostics support the proposed mechanism, but they also expose the
remaining limitation.

### 6.1 High-dimensional rare-core occupancy

| Dimension | Prior rare-core mass | Adaptive rare-core mass |
|---:|---:|---:|
| 8 | `0.00061` | **`0.00452`** |
| 16 | `0.00220` | **`0.00354`** |

At 8D the teacher's maximum rare-core count increased from `2.0` to `7.25`
particles on average, its final count from `1.5` to `6.5`, and the trained
generator's maximum count from `1.0` to `3.5`. At 16D the corresponding
teacher maximum rose from `3.5` to `5.25`, and generator maximum from `2.75`
to `4.75`.

Rare nearest-component mass error fell:

| Dimension | Prior error | Adaptive error |
|---:|---:|---:|
| 8 | `0.01247` | **`0.00917`** |
| 16 | `0.01744` | **`0.01207`** |

This shows that the rollout moves real particles into target-calibrated cores,
not merely into the diffuse bridge toward a rare center.

### 6.2 Remaining rare-mode defect

Absolute rare-core mass remains small, and binary component-core coverage did
not improve (`0.53125` for both arms over all rare targets). At 16D, bridge
mass increased by about `0.00293`, although core mass and rare-mass
calibration improved.

Therefore the experiment demonstrates partial rare-mode recovery, not full
rare-component matching. The next bottleneck is still the retention of
low-mass extreme teacher particles by the single-head neural generator.

### 6.3 Post-hoc component-conditioned moments

`analyze_component_conditioned_moments.py` closes the remaining metric-audit
item from the failure diagnosis. It assigns saved reference/output samples to
their nearest target component center and compares conditional means and
covariances. This was computed after the preregistered analysis and is a
mechanism diagnostic, not a new acceptance gate.

On rows where both candidate and fixed control had at least two samples
assigned to every component:

| Candidate / fixed | Geometric error ratio |
|---|---:|
| component mean RMSE | `0.8867` |
| component covariance relative Frobenius error | `0.8743` |
| rare-component mean error | `0.8773` |
| rare-component covariance error | `0.9058` |

The low-dimensional ratios are exactly one. At 8D, the corresponding ratios
are `0.776 / 0.803 / 0.818 / 0.879`; at 16D they are
`0.785 / 0.709 / 0.705 / 0.747`.

The candidate had enough nearest-component samples to compute all component
moments in all 32 GMM target/initialization rows. The fixed control failed in
one row. The paper comparator failed in all 16 rare-GMM rows. This finite-count
statement does not imply genuine core occupancy—the core diagnostic above
remains the stricter test.

Machine-readable report:
`adaptive_rollout_component_moments.json`  
SHA-256:
`eb0976c636277a39f3ffb201e4eacd4c1595706406dfe68d64d3cea36298295a`.

## 7. Cost result

Candidate versus fixed control:

| Ledger | Ratio |
|---|---:|
| generator forward calls | `1.0000` |
| online wall time | `1.3926` |
| projection scalar products | `1.1848` |
| sort work | `1.0678` |
| local kernel pairs | `1.4142` |

Candidate versus the previous gated hybrid:

| Ledger | Ratio |
|---|---:|
| generator forward calls | `1.0000` |
| online wall time | `1.0599` |
| projection scalar products | `1.0822` |
| sort work | `1.0671` |
| local kernel pairs | `1.0000` |

The rollout improves sample quality without additional generator calls, but it
does more non-neural transport work. It should be described as a favorable
quality/compute tradeoff, not as universally faster or cheaper.

## 8. Visual summaries

- `adaptive_rollout_confirmation_figures/quality_and_cost.png`
- `adaptive_rollout_confirmation_figures/rare_core_mechanism.png`

## 9. Reproduction

Run the regression suite:

```powershell
uv run --python 3.12 --with torch==2.7.1 --with numpy `
  --with datasketches==5.2.0 python numerics/neural_pooled_rank_tests.py
```

The exact frozen commands are recorded in
`AdaptiveRolloutConfirmationProtocol.md`. Recompute the paired analysis with:

```powershell
uv run --python 3.12 --with numpy `
  python numerics/analyze_adaptive_rollout_confirmation.py `
  --concentrated numerics/conditioned_transport_runs/20260723-161850-consumed `
  --broad numerics/conditioned_transport_runs/20260723-162201-consumed `
  --registry numerics/adaptive_rollout_confirmation_registry.json
```

Analysis SHA-256:
`3e7e37bb4070e9034da054335a1874c059f937ffe14ac4864988430badb9a043`.
