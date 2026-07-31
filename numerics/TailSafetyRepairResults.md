# Projected-Tail and Safety Repair Results

**Status:** repairs implemented and audited; neither variant promoted  
**Date:** 2026-07-23  
**Development protocol:** `TailSafetyRepairExperiment.md`  
**Analysis:** `tail_safety_repair_analysis.json`

Analysis SHA-256:
`4a7328a484d05c3931ae270f59c6b966cbb340e71f8114ef7ba55e9d2339ff8b`.

## 1. Verdict

The implementation gaps identified in the audit were repaired, but the
repaired safety and rank-aware tail variants did not improve the confirmed
dimension-adaptive rollout.

| Arm / adaptive baseline | ED2 ratio | Held-out SW1 ratio | Wins (ED2 / SW1) |
|---|---:|---:|---:|
| safety only | `1.1334` | `1.0727` | `1/16 / 1/16` |
| rank-aware balance only | `1.0076` | `1.0071` | `3/16 / 2/16` |
| safety + rank-aware balance | `1.1301` | `1.0783` | `2/16 / 1/16` |

Lower is better. No arm met the preregistered development rule for fresh
confirmation.

The confirmed `cta-exact-adaptive-rollout` remains the primary model.

## 2. Repairs completed

### 2.1 Rank-aware priority

The former displacement-only score was replaced, for the new arms, by:

```text
0.5 * percentile(displacement squared norm)
+ 0.5 * percentile(mean projected-rank extremeness).
```

Both terms are percentile normalized over the same 512-particle cohort.
The upper 10%, or 52 particles, are distributed without replacement across
the eight 64-sample student batches.

### 2.2 Exact exposure and objective accounting

Every particle is included exactly once. Since the registered batches are
equal size and no stratum is oversampled, inverse-stratum weights would all be
one; the implementation changes order rather than sample frequency.

The sample-weighted student loss is now reconstructed from per-example losses.
Tail and bulk losses are recorded independently.

### 2.3 Direct same-cohort retention

After the student micro-updates, the exact same latent cohort is reevaluated.
The ledger now records:

- teacher rare-core count;
- post-student rare-core count;
- the number of steps retaining a teacher rare-core hit; and
- teacher-to-post-student rare-core count ratio.

This closes the previous ambiguity between “teacher never arrived” and
“student lost the teacher.”

### 2.4 Corrected compute ledger

The rank-aware arms now charge:

- the teacher-rank projection;
- the teacher-rank sort;
- top-k tail selection;
- particle reordering; and
- diagnostic-only generator forward calls separately from training calls.

## 3. Verification

The official regression suite passes, including new checks for:

- finite percentile-normalized priority;
- exact rank-aware permutation;
- equal-batch tail distribution;
- ragged-batch exact exposure;
- post-student feature capture; and
- exact recombination of tail and bulk losses.

The end-to-end screen artifact is:

`conditioned_transport_runs/20260723-170940-consumed/`

Its deep audit passes and recomputes all 64 saved endpoints.
The unmodified adaptive baseline is bitwise unchanged in ED2, held-out SW1,
and training quantile RMSE relative to its earlier consumed artifact.

## 4. Why the repairs did not help

### 4.1 Tail examples are hard but not meaningfully underexposed

For the rank-aware arm, the selected tail's student loss averaged about
`3.10x` the bulk loss. The priority score therefore identifies genuinely hard
examples.

But a random 64-sample batch drawn from a 10% stratum already contains about
6.4 tail examples on average. With eight batches and 52 tail particles, exact
balancing merely changes the ordinary random variation around six or seven
tail examples per batch. It does not add examples, increase their loss weight,
or increase model capacity.

The resulting order perturbation is too weak to overcome the existing
amortization limitation.

### 4.2 Safety removes useful local correction

The safety rule reduced the local weight and worsened both primary metrics:

- 8D ED2/SW1 ratios: `1.1717 / 1.1039`;
- 16D ratios: `1.4081 / 1.1992`.

The rule guarantees only a non-increase in the frozen active-bank residual at
the current particle. That myopic condition rejects directions that can be
useful after later reranking and neural updates. The result confirms that this
surrogate is too conservative.

### 4.3 Direct retention is a real bottleneck

On rare-GMM targets, the baseline teacher's maximum rare-core count averaged
`11.0`, while the same-cohort post-student maximum averaged `7.75`. Its mean
teacher-to-post-student count ratio was about `0.436`.

Rank-aware ordering raised that ratio only to `0.449`, while final rare-core
mass fell from `0.00879` to `0.00818`. Safety and the combined arm also reduced
final rare-core mass.

Thus order regularization does not solve the retention deficit.

## 5. Dimension and family details

Rank-aware balance was the only near-neutral repair:

| Dimension | ED2 ratio | SW1 ratio |
|---:|---:|---:|
| 2 | `1.0000` | `1.0000` |
| 4 | `1.0000` | `1.0000` |
| 8 | `1.0527` | `1.0380` |
| 16 | `0.9791` | `0.9911` |

The small 16D gain was discovered on the consumed registry and is below one
percent for SW1. Selecting “balance only in 16D” now would be post-hoc tuning,
so it is not promoted.

By family, rank-aware ED2 ratios were:

- balanced GMM: `1.0141`;
- correlated Student-t: `1.0394`;
- nonlinear: `0.9929`;
- rare GMM: `0.9848`.

Held-out SW1 improved only for nonlinear (`0.9909`) and was essentially neutral
or worse elsewhere.

## 6. Cost

Relative to the adaptive baseline:

| Arm | Wall time | Projection work | Sort work | Kernel pairs |
|---|---:|---:|---:|---:|
| safety | `1.0656` | `1.2382` | `1.0389` | `1.0000` |
| rank-aware balance | `1.0719` | `1.0443` | `1.0402` | `1.0000` |
| both | `1.0956` | `1.2729` | `1.0771` | `1.0000` |

The added work is now visible rather than hidden in wall time.

## 7. Decision

Do not fresh-confirm or promote these variants. The negative result is useful:

> The current retention defect is not caused by random minibatches failing to
> see hard particles. It is caused by insufficient influence or insufficient
> generator capacity for those particles.

A materially different next intervention would need to change the objective
weight of hard particles, retain them across multiple optimization steps, or
increase generator routing/capacity. Merely redistributing them within one
pass is not enough.
