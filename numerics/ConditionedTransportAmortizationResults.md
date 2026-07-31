# Conditioned transport-then-amortize development results

**Status:** consumed-registry development complete; not confirmatory  
**Authoritative smoke artifact:**
`conditioned_transport_runs/20260722-163138-smoke/`  
**Authoritative consumed artifact:**
`conditioned_transport_runs/20260722-164154-consumed/`  
**Independent earlier KLL realizations:**
`conditioned_transport_runs/20260722-163207-consumed/` and
`conditioned_transport_runs/20260722-163801-consumed/`  
**Registry SHA-256:**
`7b4cc89a68d73784303b0e2bc417d53b2f91f02511099c88c3c8e6be535a3dc6`

## Result in one sentence

The repaired neural algorithm removed the previous 16D reversal and, on the
single consumed concentrated-initialization cell for each of 16 targets, its
KLL-plus-local arm beat the paired paper neural arm on both ED2 and held-out
SW1 in all `16/16` cases. This is strong development evidence, not a general
performance claim: the targets were inspected while the repair was designed.

## What changed

The experiment implements the diagnosis in
`ConditionedTransportAmortizationResearch.md`:

1. registered directions are retained, then complete orthogonal blocks are
   appended until projected variances determine every symmetric covariance
   coordinate with condition number at most `25`;
2. a global population of `512` particles receives one coherent PSQT step;
3. that frozen particle target is amortized through eight ordinary 64-sample
   Adam steps, restoring `160` optimizer updates under the same 20,480
   generator-example budget;
4. the optional paper field is normalized to the PSQT correction's RMS before
   entering with weight `0.25`; and
5. exact and KLL target atlases are compared behind the same interface.

The first student gradient is not heuristic. The regression suite checks to
`5.6e-17` absolute error that it equals `eta` times the neural rank-loss
gradient. Later microsteps are explicitly frozen-teacher distillation.

## Aggregate endpoints

Medians below are over the same 16 target instances, replication zero, and
concentrated initialization. Baselines come from the consumed Phase-1
artifact; revised arms come from the new consumed artifact.

| Arm | Median ED2 | Median held-out SW1 |
|---|---:|---:|
| paper neural | 0.04079 | 0.14435 |
| KLL small RSR (previous best neural atlas arm) | 0.04407 | 0.14926 |
| exact conditioned global | 0.02021 | 0.09220 |
| KLL conditioned global (final realization) | 0.02061 | 0.09482 |
| exact conditioned + local | **0.01712** | **0.08260** |
| KLL conditioned + local (final realization) | 0.01491 | 0.08074 |

In the earlier KLL realization, conditioned plus local had geometric-mean
ratios `0.3819` for ED2 and `0.5927` for held-out SW1 against paper. Ratios
below one favor the new arm. It beat paper on both metrics in `16/16` targets
in both KLL realizations; it beat previous small KLL-RSR on both in `15/16` in
the earlier realization. The exact hybrid also beat paper on both in `16/16`.

These ratios must not receive confirmatory intervals: selecting the mechanism,
condition threshold, local weight, particle step, and learning rate involved
these consumed target instances.

For the now-selected deterministic exact hybrid, the geometric-mean ratios
against paper were `0.3735` (ED2) and `0.5858` (held-out SW1), with paired wins
on both metrics in `16/16` targets.

## Dimension check

| Dimension | Paper ED2 / SW1 | KLL conditioned + local ED2 / SW1 |
|---:|---:|---:|
| 2 | 0.02135 / 0.13007 | **0.00986 / 0.07858** |
| 4 | 0.04549 / 0.15670 | **0.00982 / 0.07209** |
| 8 | 0.06414 / 0.16637 | **0.01555 / 0.07728** |
| 16 | 0.06476 / 0.13632 | **0.02684 / 0.08684** |

The old small-KLL method reversed at 16D. The revised method does not. This is
consistent with the causal direction-bank intervention: the 64 registered
16D directions sensed only `61/136` covariance coordinates. The condition
rule selected `176` or `192` directions in 16D, with measured condition
numbers between `18.92` and `24.90`. It left all lower-dimensional banks at
64 directions.

## Exact versus KLL

KLL retained the qualitative exact-atlas behavior, but Apache DataSketches KLL
uses internal randomized compaction that is not seeded by this Python API.
Three complete builds from the same target pool therefore provide an
informative repeatability check. Exact arms reproduced bit-for-bit. The KLL
hybrid aggregate medians were respectively `0.01771 / 0.08376`,
`0.01323 / 0.08068`, and `0.01491 / 0.08074`. Across pairs, median absolute
per-target differences stayed below `0.00089` (ED2) and `0.00274` (SW1).
Maximum differences were larger (`0.01909` / `0.05654`), showing
neural-trajectory amplification in one nonlinear cell.

In the earlier realization, median absolute endpoint differences between exact
and KLL were:

| Variant | ED2 difference | SW1 difference |
|---|---:|---:|
| global | 0.00096 | 0.00340 |
| global + local | 0.00047 | 0.00182 |

KLL nevertheless beat paper in every paired cell in both builds. Thus sketch
error is not the dominant aggregate bottleneck, but sketch randomness is a
real source of algorithmic variability. The exact arm must remain the
deterministic quality reference. A confirmatory KLL atlas must serialize and
hash its payloads before model training, and a later robustness study should
sample multiple independent sketch builds.

## Rare-mode behavior

The final KLL conditioned-plus-local arm retained full mode coverage on every
rare mixture in this cell. Its rare-mass errors from dimensions 2 through 16
were `0.0109`, `0.0208`, `0.0145`, and `0.0168`. Thus the endpoint gains were
not bought by dropping the registered 5% component.

The strengthened protected-tail selector compares local candidates with the
global-only candidate, so the local term cannot spend improvement supplied by
the global correction. It was genuinely active: the mean selected weight was
`0.167`, and at least one macro-step selected zero in most families. The
guarded endpoint remained better than paper in all `16/16` paired cells, but
was worse than the unguarded KLL hybrid in aggregate (`0.01680 / 0.09198`). It
is therefore a functioning conservative ablation, not the selected quality
arm.

## Work accounting

All neural arms used exactly 20,480 generator-example evaluations.

| Arm | Adam updates | Forward calls | Projection products (median) | Paper kernel pairs (median) |
|---|---:|---:|---:|---:|
| paper neural | 320 | 320 | 0 | 2,621,440 |
| previous KLL small RSR | 160 | 320 | 2,621,440 | 0 |
| KLL conditioned + local | 160 | 320 | 2,621,440 | 10,485,760 |
| KLL guarded + local | 160 | 320 | 6,553,600 | 10,485,760 |

The new arm is compute-matched only in generator examples and calls. It is not
cheaper than paper: its 512-by-512 local field uses four times as many kernel
pairs as the paper's 64-by-64 updates, and it additionally computes global
projections. CPU wall-time medians happened to be similar (`0.46 s` for the
new unguarded KLL hybrid and `0.66 s` for paper), but this small synthetic run
is not an implementation-independent efficiency claim.

## Audit and interpretation

The following are now executable and tested:

- vector-frame versus quadratic-frame separation;
- explicit failure on a rank-deficient or ill-conditioned covariance design;
- deterministic extension to a condition-certified bank;
- exact first-gradient identity;
- deterministic transport micro-updates and exact budget counts;
- zero particle step and zero local field;
- KLL transport finiteness; and
- adversarial local-field rejection by the tail guard.

The finalized raw artifact contains every row, output sample, reference sample,
serialized KLL state, state hash/index, source snapshot, registry copy, and
manifest. Its fail-closed `audit.json` records `80` unique cells, `80` outputs,
`16` references, and `1,504` validated KLL states totaling `2,304,128` bytes.

## Decision

The implementation has crossed the development threshold for a fresh test.
Freeze **exact conditioned-plus-local** as the deterministic primary and
unguarded **KLL conditioned-plus-local with pre-serialized payloads** as the
retention arm. Keep the guarded version as a safety ablation, not as the
primary: it never intervened and adds projection work. The next step is to
generate a genuinely new target registry, freeze the KLL payloads before
training, and preregister target-level paired gates before observing any model
outcomes.

Until that succeeds, the correct statement is: **we found and reproduced a
mechanism that dominates the repository's paper port on the consumed synthetic
development matrix.** We have not yet established general superiority, and we
have not compared against the paper's image benchmarks or metrics.

## Fresh confirmation update

The recommended fresh experiment is complete and passed every preregistered
exact-primary and KLL-retention gate. See
`NeuralConditionedTransportConfirmatoryProtocol.md` and
`NeuralConditionedTransportConfirmatoryResults.md`. This supersedes the
development-only uncertainty above within the scoped multidimensional
synthetic neural benchmark; it does not broaden the claim to images.
