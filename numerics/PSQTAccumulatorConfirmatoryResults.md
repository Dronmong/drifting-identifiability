# PSQT accumulator repair: sealed confirmatory result

**Status:** completed; both bounded accumulator arms passed their preregistered
promotion gates.

**Sealed artifact:**
`psqt_accumulator_confirmatory_runs/20260722-005506-confirmatory/`

**Registry SHA-256:**
`1f8196b8dc73acddec03bc6ed31f6c19c3d9236a321f84e1ed5df459ad867a54`

## Result in one sentence

On the frozen suite of 64 fresh two-dimensional target instances, persistent
pooled-rank target statistics substantially improved both historical online
PSQT and the registered `tau = 1` paper estimator under the common sample and
particle budgets. The Apache KLL quality arm passed every preregistered gate,
supporting the protocol's bounded claim of a **general low-dimensional
improvement**.

This is a result for the declared nonparametric 2D particle-generator testbed.
It is not a claim about ImageNet, encoder features, neural generators,
arbitrary dimension, every paper bandwidth, or asymptotic optimality.

## What was confirmed

The experiment compared five frozen arms:

1. the repository's paper estimator at `tau = 1`;
2. historical PSQT, which averages minibatch quantile tables;
3. PSQT using one Apache DataSketches KLL sketch per fixed projection;
4. PSQT using a reservoir of 1,024 raw 2D target samples; and
5. exact pooling as a diagnostic ceiling, ineligible for promotion.

Every primary arm received the same initial particles, ordered target stream,
evaluation sample, directions, 19,200 target observations, 64 particles, and
300-update budget within each paired trial. Inference used target instances,
not optimizer seeds, as the independent units. The registry contained eight
fresh instances from each of eight prespecified families, with five paired
streams and two initializations per target.

Lower ratios below mean that the candidate had lower error than the named
baseline.

| Candidate and endpoint | Baseline | Geometric-mean ratio | Target-bootstrap 95% CI | Targets won |
|---|---|---:|---:|---:|
| KLL, ED2 | historical PSQT | 0.337 | [0.311, 0.367] | 98.4% |
| KLL, ED2 | paper `tau = 1` | 0.076 | [0.063, 0.093] | 100.0% |
| Reservoir, ED2 | historical PSQT | 0.553 | [0.511, 0.602] | 78.1% |
| Reservoir, ED2 | paper `tau = 1` | 0.125 | [0.104, 0.152] | 100.0% |
| KLL, held-out SW1 | historical PSQT | 0.519 | [0.500, 0.539] | 100.0% |
| KLL, held-out SW1 | paper `tau = 1` | 0.289 | [0.260, 0.322] | 100.0% |
| Reservoir, held-out SW1 | historical PSQT | 0.676 | [0.651, 0.703] | 100.0% |

Equivalently, the KLL arm reduced geometric-mean ED2 by about 66% relative to
historical PSQT and 92% relative to the registered paper arm. Its held-out SW1
fell by about 48% and 71%, respectively. These are paired finite-benchmark
effects, not population guarantees.

As a diagnostic, exact pooling achieved an ED2 ratio of 0.318 versus
historical PSQT, compared with KLL's 0.337. Thus the bounded KLL arm was only
about 6% above the unbounded exact-pooled ceiling in geometric-mean ED2, while
using about 15.5% of its persistent-state ledger. This comparison was not a
selection endpoint.

## Robustness across target families

KLL beat historical PSQT in geometric-mean ED2 in every prespecified family:

| Family | KLL / historical PSQT ED2 | KLL / paper ED2 |
|---|---:|---:|
| Gaussian mixtures | 0.356 | 0.128 |
| Disconnected non-Gaussian | 0.140 | 0.062 |
| Rare modes | 0.193 | 0.055 |
| Correlated unimodal | 0.655 | 0.009 |
| Curved connected | 0.482 | 0.096 |
| Multiple curves | 0.487 | 0.150 |
| Skewed/heavy-tailed | 0.782 | 0.245 |
| Dependence traps | 0.144 | 0.083 |

The weakest KLL family result against historical PSQT was the skew/heavy-tail
family at 0.782, still below the preregistered no-family-regression threshold
of 1.10. Both KLL and the reservoir recovered every evaluated 5% and 10% rare
mode; historical PSQT also recovered all of them, while the paper arm recovered
72%. KLL's median excess bridge occupancy was exactly one particle (1/64),
meeting its gate.

## Cost and state

The quality and efficiency arms occupy different Pareto points:

| Arm | Median persistent bytes | Median wall time per paired trial |
|---|---:|---:|
| Paper `tau = 1` | 1,024 | 0.0796 s |
| Historical PSQT | 18,440 | 0.1062 s |
| KLL-k128 PSQT | 48,000 | 0.0278 s |
| Reservoir-1024 PSQT | 18,448 | 0.0312 s |
| Exact pooled ceiling | 309,248 | 0.0275 s |

The KLL arm uses roughly 2.6 times the persistent state of historical PSQT,
so it should be presented as the quality arm, not as a state-saving method.
Reservoir-1024 stays within 0.1% of historical PSQT's persistent-byte ledger
while improving both quality metrics, which is why it separately passed the
efficiency promotion. Wall time favors both new arms on this machine and in
these NumPy/compiled-sketch implementations, but wall time remains
implementation- and hardware-specific; the operation and memory ledgers in
`rows.csv` are the portable records.

## Preregistered gate outcome

All KLL quality gates passed:

- ED2 effect and target-level interval versus historical PSQT;
- ED2 effect and target-level interval versus the paper arm;
- held-out SW1 improvement versus both baselines;
- target win fraction;
- family robustness;
- zero divergence;
- 5%/10% rare-mode recovery; and
- bridge-occupancy control.

All Reservoir-1024 efficiency gates also passed, including its state and
wall-time gates. No arm diverged in any of the 3,200 trials. Exact pooling was
reported only as a ceiling and was not eligible for selection.

## Accumulator validation

The promoted quality implementation uses Apache DataSketches 5.2.0 with
`k = 128`, run under Python 3.12. The independent stress audit covered
Gaussian, Student-t, log-normal, separated mixtures, rare mixtures, repeated
values, adversarial orders, stream chunking, and sketch merging.

- official normalized-rank error bound: 0.02052;
- maximum observed rank error: 0.01672;
- 95th percentile of per-case maximum rank error: 0.01152;
- median maximum rank error: 0.00651;
- state remained bounded through a 100,000-observation stress stream; and
- monotonicity, observed-support, exact-mode, accounting, merge, and
  serialization/replay checks passed.

Apache's Python KLL binding does not expose a compaction RNG seed. Therefore
the sealed artifact preserves every realized serialized state rather than
claiming seed-level bit-for-bit regeneration.

## Artifact audit

The post-run audit verified:

- all 13 frozen source/dependency-audit hashes still matched;
- the registry hash matched its pre-run sidecar and freeze manifest;
- 64 targets, 3,200 unique trial cells, and 320 target-arm aggregates existed;
- all output arrays were finite and had shape `(3200, 64, 2)`;
- all reference arrays were finite and had shape `(64, 4096, 2)`;
- the KLL binary contained 20,480 contiguous, hash-matching direction states;
- the states covered 640 KLL trials and all 32 directions per trial;
- every deserialized sketch reported exactly 19,200 observations; and
- the run recorded zero missing trials and zero numerical divergences.

The artifact also stores the frozen registry and source snapshots, bootstrap
draws, raw rows, target aggregates, KLL state index and binary, particle and
reference arrays, manifest, summary, and one prespecified visualization per
family.

## Honest conclusion and next boundary

The repaired target-statistic mechanism is no longer merely a promising
development result. It survived a separately generated, preregistered,
target-level confirmatory experiment and beat both registered baselines across
all eight low-dimensional families. The central mechanism is also clear:
preserving pooled projected ranks avoids the quantile bias introduced by
averaging minibatch quantile functions.

The next scientifically useful test is not another adjustment on these 64
sealed targets. These targets are now consumed as confirmatory evidence. A
new protocol and new targets are required for any follow-up, with the most
valuable next boundary being scaling in dimension and/or evaluation on frozen
real feature distributions.
