# Neural pooled-rank Phase-1 standard development results

**Status:** standard development complete; not confirmatory  
**Merged artifact:**
`neural_pooled_rank_runs/20260722-160125-standard-merged/`  
**Registry SHA-256:**
`7b4cc89a68d73784303b0e2bc417d53b2f91f02511099c88c3c8e6be535a3dc6`

## Audit outcome

The registered standard matrix completed all `672` rows:

```text
16 targets x 3 replications x 2 initializations x 7 arms = 672
```

Eight two-target CPU shards were merged only after checking complete shard
coverage, registry equality, shard-file hashes, KLL payload hashes, disjoint
output and reference keys, the exact target set, row counts, and matched
generator-example budgets. There were no divergences. The merged artifact
contains every row, output, reference, KLL state and index, source snapshot,
source-shard hash, and manifest. The full Phase-0 regression suite also ran in
every shard.

This is a development result. Learning rates were chosen on the four consumed
smoke targets described in `NeuralPooledRankOptimizationScreenResults.md`.
Consequently neither the smoke nor this registered development matrix is an
independent confirmation of the selected design.

## Overall endpoints

These are medians over all 96 replication/initialization cells for each arm.

| Arm | Median ED2 | Median held-out SW1 | Divergences |
|---|---:|---:|---:|
| paper neural | **0.04631** | 0.15257 | 0 |
| minibatch SW | 0.05025 | 0.15228 | 0 |
| exact-atlas large RSR | 0.05928 | 0.15971 | 0 |
| KLL-atlas large RSR | 0.05907 | 0.16105 | 0 |
| KLL-atlas small RSR | 0.05060 | **0.14766** | 0 |
| KLL-plus-paper hybrid | 0.05699 | 0.15848 | 0 |
| exact free-particle ceiling | 0.00436 | 0.04720 | 0 |

Small-KLL has the best overall held-out sliced-Wasserstein median, about `3.2%`
below paper and `3.0%` below minibatch SW. It does not win ED2: its median is
about `9.3%` above paper and `0.7%` above minibatch SW. The correct conclusion
is therefore competitive, metric-dependent performance—not general
outperformance.

## Target-level paired assessment

Optimizer replications and the two initializations are not independent target
problems. For the paired comparison below, each arm is first reduced to its
median within each of the 16 target instances.

| Small-KLL comparison | ED2 wins | Held-out SW1 wins | Wins both | Median ED2 difference | Median SW1 difference |
|---|---:|---:|---:|---:|---:|
| versus paper | 7/16 | 8/16 | 7/16 | +0.00445 | -0.00354 |
| versus minibatch SW | 9/16 | 9/16 | 8/16 | -0.00161 | -0.00280 |
| versus exact large RSR | 12/16 | 12/16 | 12/16 | -0.00712 | -0.01367 |
| versus KLL large RSR | 12/16 | 12/16 | 12/16 | -0.00748 | -0.01454 |

Negative differences favor small-KLL. Exploratory target bootstrap intervals
for the mean small-KLL improvement include zero against paper, minibatch SW,
and large RSR on both primary metrics. With only 16 heterogeneous targets, the
study does not establish a broad population-level win.

## Where persistent target ranks helped

The clearest signal is rare-mode preservation. On the four registered 5%
rare-mixture targets, medians over all cells were:

| Arm | Mode coverage | L1 mode-mass error | Rare-mass error |
|---|---:|---:|---:|
| paper neural | 0.50 | 0.1000 | 0.0500 |
| minibatch SW | 1.00 | 0.0420 | 0.0210 |
| exact large RSR | 0.75 | 0.0425 | 0.0212 |
| KLL large RSR | 1.00 | 0.0434 | 0.0217 |
| **KLL small RSR** | **1.00** | **0.0283** | **0.0142** |
| KLL-plus-paper hybrid | 0.50 | 0.0702 | 0.0351 |

Small-KLL reduces median rare-mass error by about `71.7%` versus paper and
`32.6%` versus minibatch SW. This matches the intended mechanism: an immutable
target atlas retains low-probability quantiles that a fresh target minibatch
can omit. It is the strongest reason to continue this line, but it is a
specific occupancy result rather than a general quality win.

Family-level target medians tell the same story. Small-KLL is strongest on
correlated heavy tails and rare mixtures, competitive on balanced mixtures,
and not consistently better on nonlinear targets.

## Dimension and hybrid behavior

Raw cell medians by dimension are:

| Dimension | Paper ED2/SW1 | Minibatch SW ED2/SW1 | Small-KLL ED2/SW1 |
|---:|---:|---:|---:|
| 2 | 0.0285 / 0.1640 | 0.0298 / 0.1583 | **0.0249 / 0.1383** |
| 4 | **0.0405** / 0.1576 | 0.0428 / 0.1620 | 0.0420 / **0.1481** |
| 8 | 0.0519 / 0.1476 | 0.0527 / 0.1377 | **0.0471 / 0.1342** |
| 16 | **0.0714 / 0.1497** | 0.0908 / 0.1590 | 0.1023 / 0.1679 |

The small-KLL benefit survives through 8D but reverses at 16D. Interestingly,
the local-plus-global hybrid is strongest only in 16D (`0.0726 / 0.1360`),
where it approximately retains paper's ED2 and has the best held-out SW1.
Because its learning-rate screen was nonmonotone, it adds kernel and target
cost, and it loses overall, this is a redesign clue rather than a promoted
result.

Both concentrated and broad initializations show nearly the same ordering, so
the overall result is not caused by one initialization regime.

## Mechanism diagnosis

Exact and KLL large-RSR are nearly indistinguishable. KLL approximation is not
the limiting factor. The exact free-particle ceiling is dramatically better
than every neural arm, showing that the fixed directions and exact target
quantiles contain a strong transport signal.

The remaining bottleneck is neural amortization and optimizer time. Under the
matched generator-example budget:

| Arm | Adam updates | Generator forward calls | Unique latent samples |
|---|---:|---:|---:|
| paper/minibatch SW | 320 | 320 | 20,480 |
| exact/KLL large RSR | 20 | 320 | 10,240 |
| KLL small RSR | 160 | 320 | 10,240 |

Large RSR pays two evaluations per latent and compresses the budget into only
20 parameter updates. Learning-rate repair removed the gross failure but did
not make those 20 updates superior to 160 small-RSR updates. A larger rank
population is therefore not the causal winner in this neural setting.

Small-KLL uses the same generator-example evaluations and forward-call count
as the baselines. Its median training-only wall time was lower than minibatch
SW in this CPU run and its sorting proxy was smaller because the target atlas
is precomputed, but wall time excludes evaluation and is affected by parallel
contention. It must not be advertised as categorically cheaper. It also sees
half as many unique latent samples because of replay.

## Go/no-go decision

The original Phase-1 gate is **not fully passed**:

- exact large-RSR does not improve over minibatch SW;
- KLL faithfully retains exact-atlas behavior;
- small RSR is better than large RSR, contrary to the proposed large-rank
  mechanism;
- improvements occur beyond 2D but reverse in aggregate at 16D;
- the free-particle ceiling confirms large remaining headroom; and
- the hybrid does not win overall, although its 16D behavior is informative.

Accordingly, do not yet proceed to an image claim or present neural KLL-PSQT as
generally better than the paper. The next development cycle should preserve
the demonstrated persistent-atlas rare-mode benefit while repairing the
high-dimensional amortization failure. The most defensible next experiment is
a dimension/conditioning-aware local-global schedule with frequent small-RSR
updates, compared against equally tuned paper and minibatch baselines on new
development targets. Frozen real-feature work becomes warranted only after
that redesign wins a target-level paired gate rather than one aggregate
median.

## Successor checkpoint

The direction-bank and optimizer-time failures identified above were later
repaired by conditioned transport-then-amortize. See
`ConditionedTransportAmortizationResearch.md` for the derivation and
`ConditionedTransportAmortizationResults.md` for the audited consumed-registry
development result. This note does not retroactively change Phase 1's failed
gate; the successor is a new algorithm and still needs fresh confirmation.
