# Neural pooled-rank optimization-repair screen

**Status:** completed development tuning; not confirmatory  
**Artifact:** `neural_pooled_rank_optimization_runs/20260722-154549-lr-screen/`  
**Registry SHA-256:**
`7b4cc89a68d73784303b0e2bc417d53b2f91f02511099c88c3c8e6be535a3dc6`

## Why this screen was necessary

The initial smoke run matched generator-example evaluations but not optimizer
steps. Ordinary arms received 64 Adam updates, exact/KLL large-population RSR
received 8, and small-population RSR received 32. At the common `1e-3` rate,
endpoint quality followed update count. Exact and KLL large-RSR results were
nearly identical and an exact free-particle ceiling was excellent, localizing
the problem to neural optimization rather than sketch accuracy or an
impossible target table.

The repair screen reused only those four already-consumed smoke targets. It
paired initialization, latent stream, target data, directions, and evaluation
samples while changing Adam's learning rate. It tuned every neural arm,
including both paper and minibatch-SW baselines. The artifact contains 128
rows, all outputs and references, exact source snapshots, a dirty-tree
manifest, and all 256 realized Apache KLL states. No run diverged.

## Median screen curves

| Arm | Multiplier | Median ED2 | Median held-out SW1 |
|---|---:|---:|---:|
| paper neural | 1 | 0.2366 | 0.3924 |
| paper neural | 8 | **0.1031** | 0.2360 |
| paper neural | 16 | 0.1207 | **0.2037** |
| paper neural | 32 | 0.1417 | 0.2205 |
| minibatch SW | 1 | 0.1613 | 0.2577 |
| minibatch SW | 8 | 0.0871 | 0.2044 |
| minibatch SW | 16 | 0.1016 | **0.1935** |
| minibatch SW | 32 | **0.0846** | 0.1997 |
| exact-atlas large RSR | 8 | 0.2520 | 0.3192 |
| exact-atlas large RSR | 16 | 0.2030 | 0.2962 |
| exact-atlas large RSR | 32 | **0.1526** | **0.2314** |
| exact-atlas large RSR | 64 | 0.1586 | 0.2847 |
| KLL-atlas large RSR | 8 | 0.2530 | 0.3213 |
| KLL-atlas large RSR | 16 | 0.2036 | 0.2990 |
| KLL-atlas large RSR | 32 | **0.1459** | **0.2352** |
| KLL-atlas large RSR | 64 | 0.1763 | 0.2903 |
| KLL-atlas small RSR | 4 | 0.1356 | 0.2257 |
| KLL-atlas small RSR | 8 | 0.1175 | 0.2135 |
| KLL-atlas small RSR | 16 | **0.0984** | **0.2053** |
| KLL-atlas small RSR | 32 | 0.1468 | 0.2468 |
| KLL-plus-paper hybrid | 8 | 0.2379 | 0.3163 |
| KLL-plus-paper hybrid | 16 | 0.3180 | 0.4189 |
| KLL-plus-paper hybrid | 32 | **0.1639** | **0.2585** |
| KLL-plus-paper hybrid | 64 | 0.1843 | 0.2927 |

Bold values mark the best displayed endpoint for that arm and metric, not a
statistical winner. The complete grid also includes smaller baseline rates.

## What was learned

The optimization diagnosis was correct. Exact large-RSR ED2 improved from
`0.9148` at the original rate to `0.1526` at `32x`; small KLL improved from
`0.2740` to `0.0984` at `16x`. Both searches have an observed turning point,
so the selected rates are not merely the largest tried values.

Fair tuning also prevents an inflated pooled-rank claim. On the four smoke
targets, the selected small-KLL arm beats both selected baselines on ED2 and
held-out SW1 for the 4D rare mixture and 16D nonlinear law, is close in 8D,
and loses in 2D. Its median is competitive but not uniformly superior. Large
RSR remains worse under the matched generator-example budget, so a larger
current population has not yet justified its extra sorting and replay cost.

The hybrid response is nonmonotone and does not improve the pure arms. It is
retained in the standard study as a diagnostic, not as the leading method.

## Frozen next step

The arm-level rates now recorded in `NeuralPooledRankPhase1Protocol.md` are
fixed before the standard development run. That run uses all 16 registered
targets, both initializations, and three replications. The smoke targets have
served their tuning purpose and cannot be treated as new evidence. A later
confirmatory claim still requires a fresh registry after the architecture and
hyperparameters are frozen.
