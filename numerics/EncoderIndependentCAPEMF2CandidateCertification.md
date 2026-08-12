# CAP-EMF-2: the admission failure and the candidate certification path

Status: **in progress.** Production admission returned `NO_GO`. This document
records why, what in that verdict is real, two defects found while
investigating, and the free local experiment now deciding whether the campaign
can proceed at all.

Spend to date is about $15 of the $60 envelope. Nothing further has been spent
while this question is open.

## 1. Where the campaign stopped

The paid pipeline runs `admission` before anything else. It audits the
finite-difference quotient used by the direct-`x` Euler Mean Flow objective
against an exact forward-mode JVP of the same network, on the preserved
CAP-EMF-1 checkpoint, across seven `(t, r)` strata × three input sources ×
three repeats.

`local_1000_d0002_fp32` — the predeclared production candidate, embedding scale
1000 with δ = 2·10⁻⁴ — failed one stratum:

| stratum | t | r | coefficient | result |
| --- | --- | --- | --- | --- |
| `exact_inference` | 1.0 | 0.0 | 10.0 | pass |
| `near_inference` | 0.995 | 0.0 | 9.9 | pass |
| `large_coefficient` | 0.98 | 0.02 | 9.4 | pass |
| **`interior_high`** | **0.85** | **0.10** | **6.4** | **fail 0/9** |
| `interior_mid` | 0.60 | 0.20 | 2.0 | pass |
| `low_t_weighted` | 0.03 | 0.0 | ~0.01 | pass |
| `below_floor_weighted` | 0.01 | 0.0 | ~0.001 | pass |

The failure is systematic, not noise: `numerical_admission` sets
`torch.use_deterministic_algorithms(True)`, and the nine failing records are
three repeats × three *different* input sources. The same stratum fails on real
CIFAR-10, Gaussian noise and a checkerboard.

It is also not a corner. Sampling the production `ordered_uniform` triangle
shows `t ∈ [0.7, 0.95] ∧ r ∈ [0.05, 0.20]` carries 3.7% of all draws — roughly
7.5% of the rows that actually bear a quotient, since about half of each batch
is diagonal and carries none. The probed point itself sits in a band holding
1.00% of training mass, the same share as `interior_mid`, which passes. This is
bulk sampled by the objective, not an unreachable singularity.

## 2. A defect in the audit, and what it does not excuse

While reviewing the failure I found a bug I introduced when separating the
denominator floors.

One constant, `emf_denominator_floor = 0.02`, had been serving three roles: the
Euler divisor (clamps `t`), the `1/t²` loss weight (clamps `t`), and the
correction coefficient's denominator (clamps `r`). Only the third needed
raising, so `resolved_coefficient_floor` was added and set to `0.10` for the
ordered arms. Training was updated. **`numerical_admission` was not** — both
`audit_stratum` and `audit_mixed_batch` kept clamping `r` with the inherited
0.02.

Wherever `r < 0.10` this inflates the coefficient fivefold. At
`exact_inference` the audit assembled its target with a coefficient of 50.0
instead of the 10.0 production actually trains on; at `large_coefficient`,
47.0 instead of 9.4. Those strata were being graded roughly five times harsher
than the paid run — an audit of an objective nobody runs.

Both sites are now fixed
([numerical_admission.py](numerics/encoder_independent_drifting/stage_cap2/numerical_admission.py)),
with a parametrized regression test asserting the coefficient matches the
resolved floor and *differs* from the inherited one.

**This does not rescue the verdict.** At `interior_high`, `r = 0.10` — exactly
the coefficient floor. `max(0.10, 0.02)` and `max(0.10, 0.10)` are the same
number, so the coefficient there was 6.373 under both the buggy and the fixed
code. The one stratum that failed is the one stratum the bug could not touch.
The fix can only *relax* the strata that already passed.

The `NO_GO` is therefore real, and re-running admission for
`local_1000_d0002_fp32` with the fix would waste money to reproduce it.

## 3. A sealing gap in the post-hoc EMA

`runpod_pipeline.sh` (a sealed dependency) invokes
`stage_cap2.posthoc_ema` in its `posthoc-ema` phase, but `posthoc_ema.py` was
absent from `_DEPENDENCIES` in
[artifacts.py](numerics/encoder_independent_drifting/stage_cap2/artifacts.py).

Every CAP2 artifact embeds the whole `source_manifest()` as `source_sha256`,
and consumers compare it for strict dict equality against the live manifest.
An unsealed module inside that pipeline means the code synthesizing the
*reported* EMA result could have been edited after the foundation's numbers
were known without changing a single recorded hash. Now added.

Because both changes touch sealed files, every artifact currently on the
volume — gate calibration, sampler audit, positive control, metric
calibration, numerical admission — is stale and must be regenerated on the next
pod. That is the known ~2.5 h evidence pass, not new work, and it is unavoidable
the moment any manifest file changes.

## 3b. The two shell scripts were not line-ending pinned

`.gitattributes` pins `eol=lf` for the CAP2 Python modules, JSON, SHA sidecars,
text requirements and the CAP markdown, with the stated reason that
authorization records "hash exact source and artifact bytes" and must stay
"LF-stable across Windows and Unix checkouts so the recorded source manifests
and SHA sidecars remain portable."

It did not cover `*.sh`. Both `runpod_bootstrap.sh` and `runpod_pipeline.sh`
are declared `_DEPENDENCIES` and are therefore hashed into every source
manifest, so under `core.autocrlf=true` a Windows working copy computes a
different SHA for byte-identical content than the Linux pod does. The committed
blobs were already LF, which is why deployment worked; the exposure was to
anything computing a manifest locally. The rule now covers them, and the
working copies were verified to contain no CR bytes, so no content changed.

## 4. Why the remaining candidate cannot be certified the usual way

Three numerical candidates are registered. `legacy_1000_d01` is the documented
historical failure and may never be promoted. `local_1000_d0002_fp32` has now
failed. That leaves `smooth_100_d001_fp32` — embedding scale 100 with
δ = 10⁻³, which bounds the maximum phase step at 0.1 radian instead of 0.2
while taking a *larger* finite difference.

There is a mechanical reason to expect it to condition better. The two error
sources trade against each other: truncation error grows with δ, roundoff error
grows as ε/δ. In fp32, √ε ≈ 3.5·10⁻⁴ is the crossover for a unit-scaled
function. `local_1000_d0002_fp32` sits at δ = 2·10⁻⁴, *below* the crossover and
therefore roundoff-dominated; `smooth_100_d001_fp32` sits above it, but at a
tenfold smaller embedding scale the function varies far more slowly in phase,
so its truncation term is roughly four times smaller as well. It should improve
both terms at once.

It also cannot be admitted from the preserved checkpoint. `run_admission`
refuses outright:

> candidate embedding scale differs from the trained checkpoint;
> run a short model trained with that embedding before admission

That guard is correct — the CAP-EMF-1 weights were trained with a scale-1000
embedding, and evaluating them under a scale-100 embedding measures a different
function. The registry says the same thing in its own note: the candidate
"requires a short trained-model audit" and "cannot be certified from the
historical checkpoint alone."

The pipeline offers no route to that audit. `run_screen` takes its candidate
from the preflight artifact, and the preflight requires admission to have
passed — so training at scale 100 requires an admission that cannot be
performed until a scale-100 model exists. The protocol predeclared the remedy
but never built it.

### The certification path, now built and verified

The missing step is small, and it does not require modifying anything sealed.
The diagnostic below can write its trained state in exactly the layout
`numerical_admission._load_checkpoint` expects — raw `state_dict` plus the
`profile` that produced it — so the **real gate runs on it unmodified**:

```
# 1. produce a scale-100 model (no preflight needed; this is not a screen)
python -m numerics.cap2_candidate_diagnostic \
    --candidates smooth_100_d001_fp32 --updates 50000 --rungs 1 \
    --device cuda --data-root "$DATA_ROOT" \
    --save-checkpoint-dir "$WORKSPACE/candidate_audit" \
    --out "$WORKSPACE/candidate_audit/diagnostic.json"

# 2. certify it with the production gate, hardware-bound, on the paid GPU
python -m numerics.encoder_independent_drifting.stage_cap2.numerical_admission \
    --checkpoint "$WORKSPACE/candidate_audit/cap2_ordered_uniform_smooth_100_d001_fp32_step50000_raw.pt" \
    --candidate smooth_100_d001_fp32 --device cuda --batch 4 --repeats 3 \
    --data-root "$DATA_ROOT" --expected-gpu-name "$EXPECTED_GPU" \
    --include-gradient --out "$GATES/production/numerical_admission.json"
```

Step 2 is the ordinary sealed gate: manifest-bound, hardware-bound, production
GPU, full 7×3×3 matrix with gradients. Only step 1 is new, and it produces a
checkpoint rather than a verdict, so it cannot launder anything past the gate.

A round-trip check confirms the contract holds: the written checkpoint loads,
reports `scalar_embedding_scale = 100.0`, satisfies the candidate scale guard —
and the historical scale-1000 checkpoint is still correctly refused under the
scale-100 candidate.

## 5. The diagnostic

[cap2_candidate_diagnostic.py](numerics/cap2_candidate_diagnostic.py) builds
that missing step, and runs it locally on an RTX 4050 for free rather than on a
rented A40.

For each candidate it constructs the *production* architecture at that
candidate's embedding scale, trains a matched short run — identical seed, data
order and horizon, with only `embedding_scale` and `delta` differing — and
measures every audit stratum with `numerical_admission.audit_stratum`, the same
measurement code the real gate uses, so the rows are directly comparable to a
production admission record.

Two design points matter.

**It measures a ladder, not a point.** A 200-update calibration run failed
`exact_inference` and `large_coefficient` while *passing* `interior_high` —
the exact inverse of the 650k production checkpoint's pattern. Conditioning
clearly moves as the learned function sharpens, so a single short-horizon
measurement would be misleading. Measuring at four checkpoints inside one
training run costs nothing extra and shows whether the gap between candidates
widens or closes with training, which is the part that extrapolates.

**It refuses to be evidence.** The module lives outside `stage_cap2` and is
absent from the sealed manifest, so no preflight or gate can bind to it; it
writes `"promoting": false`, carries no hardware binding, and runs on the wrong
GPU at a fraction of the production horizon. It answers one question — is there
any prospect here worth paying for — and nothing else.

A smoke path confirmed the tooling reproduces a known trap independently: on an
untrained model every stratum reports `q_cos = 0.0000` with `a_cos = 1.00000`,
because the zero-initialized output path makes both the quotient and the exact
derivative identically zero. That is the CAP-EMF-1 failure mode — a finite
difference validated against nothing — and it is why the registry insists on a
*trained* model. It also shows the gate's `quotient_cosine` check catches what
the assembled-target check alone would miss.

## 6. Results

Matched run: 2,000 updates, rungs at 500/1000/1500/2000, three repeats × three
sources × four rows per stratum (9 records per stratum per rung), gradients
included, RTX 4050, production numerical mode.

### `local_1000_d0002_fp32` (scale 1000, δ = 2·10⁻⁴, 0.2 rad)

Strata failing, by rung: **5 → 2 → 2 → 1**.

| stratum | 500 | 1000 | 1500 | 2000 |
| --- | --- | --- | --- | --- |
| `exact_inference` | 0/9 | 9/9 | 9/9 | 9/9 |
| `near_inference` | 9/9 | 9/9 | 9/9 | 9/9 |
| `large_coefficient` | 0/9 | 0/9 | 0/9 | **0/9** |
| `interior_high` | 0/9 | 9/9 | 9/9 | 9/9 |
| `interior_mid` | 0/9 | 0/9 | 9/9 | 9/9 |
| `low_t_weighted` | 7/9 | 9/9 | 6/9 | 9/9 |
| `below_floor_weighted` | 9/9 | 9/9 | 9/9 | 9/9 |

Everything converges to passing except `large_coefficient` (t = 0.98, r = 0.02),
which fails at every rung. At the last rung its quotient cosine is 0.9999 and
its quotient RMS 0.1093 against a 0.15 limit, so the binding constraint is
`assembled_target_relative_rms` marginally over its 0.10 limit.

### Two observations that constrain how far this generalizes

**Conditioning is strongly non-monotonic in training.** `interior_high` fails
0/9 at step 500 with a quotient cosine of 0.9028, then passes 9/9 at step 1000
with 0.9997. The step-500 agreement with the production failure is coincidence,
not a reproduction of the same mechanism. A single short-horizon measurement
would have misled in either direction; this is the reason the ladder exists.

**The diagnostic is a pessimistic proxy, quantifiably so.** `large_coefficient`
fails at all four rungs here, but *passed* in the 650k production audit — and
passed there under the buggy fivefold-inflated coefficient (47 rather than 9.4).
A 650k model passes that stratum under a five-times harsher test than a
1,500-update model fails it under the correct one. Training improves
conditioning by more than the defect being measured.

It follows that this run cannot predict which stratum fails at 750k, and a
short-horizon failure is not evidence of a production failure. Its only valid
use is the matched head-to-head below.

**The failures are magnitude errors, not directional ones.** Across every
failing stratum the binding checks are `quotient_relative_rms`,
`assembled_target_relative_rms` and `assembled_target_cosine`; no gradient check
binds anywhere (gradient cosine 0.992, relative L2 0.171, norm ratio 0.873 at
the worst row inspected). The finite difference points very nearly the right way
and is the wrong size — the signature of a truncation/roundoff imbalance, which
is precisely the axis the two candidates differ on.

### `smooth_100_d001_fp32` (scale 100, δ = 10⁻³, 0.1 rad)

| stratum | 500 | 1000 | 1500 | 2000 |
| --- | --- | --- | --- | --- |
| `exact_inference` | 9/9 | 9/9 | 9/9 | **0/9** |
| `near_inference` | 9/9 | 9/9 | 9/9 | 9/9 |
| `large_coefficient` | 9/9 | 2/9 | 9/9 | 9/9 |
| `interior_high` | 9/9 | 9/9 | 9/9 | 9/9 |
| `interior_mid` | 9/9 | 9/9 | 9/9 | 9/9 |
| `low_t_weighted` | 9/9 | 9/9 | 9/9 | 9/9 |
| `below_floor_weighted` | 9/9 | 9/9 | 9/9 | 7/9 |

### Head to head

Rows passing, out of 63 per rung:

| rung | `local_1000` | `smooth_100` |
| --- | --- | --- |
| 500 | 25/63 | **63/63** |
| 1000 | 45/63 | **56/63** |
| 1500 | 51/63 | **63/63** |
| 2000 | **54/63** | 52/63 |

The scale-100 candidate wins every rung at which its training was healthy, and
is the only candidate that ever sweeps all seven strata — it does so twice. The
scale-1000 candidate never passes `large_coefficient` at any rung.

On `interior_high`, the stratum that failed production, the worst-row quotient
RMS is 0.0010 / 0.0147 / 0.0220 / 0.0268 for `smooth_100` against 0.4385 /
0.0614 / 0.0714 / 0.0680 for `local_1000` — a 440× advantage at the first rung
and roughly 3–4× thereafter, against a 0.15 limit.

This is the direction the truncation/roundoff argument predicted, and the
agreement between an a-priori mechanism and the measurement is most of why the
result is worth acting on.

### Why: the scale-1000 model learns a far sharper function

The magnitude of the exact JVP at `interior_high` — how fast the network's
output actually moves along the characteristic — separates the two completely:

| rung | `local_1000` | `smooth_100` |
| --- | --- | --- |
| 500 | 0.127 | 0.179 |
| 1000 | 0.315 | 0.080 |
| 1500 | 0.837 | 0.225 |
| 2000 | 3.541 | 0.339 |

At scale 1000 the derivative magnitude grows 28× across training; at scale 100
it stays essentially flat (1.9×). A 1000× embedding makes the network's
dependence on `(t, r)` high-frequency, and a finite difference cannot track a
function that is sharpening that fast. That is the mechanism behind the whole
failure, and it is an argument about the *embedding*, not about δ.

**This cuts both ways and the writeup should not pretend otherwise.** A flatter
function is easier to differentiate numerically, but "easier to differentiate"
is not "better model". Part of `smooth_100`'s advantage may be that it simply
learns a weaker time-dependence, which would be underfitting rather than a win.
Nothing measured here distinguishes those, because conditioning is not quality.
Only FID on a trained foundation settles it.

### The scale-100 arm collapsed between 1,500 and 2,000 updates

| step | `local_1000` moment / rank / HH | `smooth_100` moment / rank / HH |
| --- | --- | --- |
| 500 | 0.639 / 80.9 / 9.94 | 0.699 / 79.5 / 9.33 |
| 1000 | 0.792 / 84.9 / 13.13 | 0.795 / 86.8 / 11.13 |
| 1500 | 0.846 / 79.3 / 9.47 | 0.796 / 69.3 / 6.17 |
| 2000 | 0.954 / 71.2 / 8.26 | **0.311 / 42.2 / 0.583** |

The scale-1000 arm stayed healthy throughout; the scale-100 arm's second moment
fell to 0.311 and its Haar HH ratio to 0.583. Its entire step-2000 deficit is
downstream of that collapse — `exact_inference` fails five checks at once
there, including `quotient_cosine` at 0.9676, having passed comfortably at
every earlier rung.

This is a genuine warning, not a footnote: a candidate that passes the
numerical gate and then diverges is not a viable foundation, and a 750k paid
run that collapses at 600k would burn the whole budget.

Two things temper it, and one test resolves it. The diagnostic compresses
warmup to 200 updates where production uses 5,000, which is an independent and
plausible cause of instability; and this is a single seed.

### Second seed: the collapse did not reproduce

A second scale-100 run (seed 902, eight rungs at 250-update spacing for finer
resolution) trained cleanly to 2,000 updates:

| step | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| second moment | 0.397 | 0.663 | 0.786 | 0.813 | 0.839 | 0.835 | 0.849 | 0.718 |
| rank ratio | 75.8 | 90.9 | 98.2 | 94.2 | 94.5 | 90.6 | 83.3 | 65.4 |
| Haar HH | 4.91 | 11.65 | 16.04 | 13.55 | 14.87 | 10.90 | 10.06 | 3.73 |

and swept **63/63 rows at the final rung, with no failing stratum at all**.
`interior_high` and `large_coefficient` passed 9/9 at *every one of the eight
rungs* — the latter being the stratum `local_1000` never passed once.

Its exact-JVP magnitude grew 0.128 → 0.291 (2.3×), matching seed 901's 1.9× and
confirming the flat-function mechanism reproduces across seeds.

So the collapse is **one run in two**, not a property of the candidate.

### A correction worth recording

These health ratios are all two-sided gates targeting 1.0 (second moment
[0.80, 1.25], rank ratio [0.80, 1.50], Haar HH [0.50, 1.75]). At a 2,000-update
horizon both candidates sit at rank and HH ratios in the *tens*, far above those
ceilings — so a falling HH or rank is **convergence toward real statistics, not
deterioration**. An initial collapse heuristic keyed on falling HH, which
classifies a healthy run as a failure; seed 902's HH fell 16.04 → 3.73 while the
run was demonstrably fine. Collapse is now judged on second-moment ratio against
the gate floor, which classifies all three measured traces correctly:

| run | best | final | verdict |
| --- | --- | --- | --- |
| `local_1000` seed 901 | 0.954 | 0.954 | healthy |
| `smooth_100` seed 901 | 0.796 | 0.311 | **collapsed** |
| `smooth_100` seed 902 | 0.849 | 0.718 | healthy |

The protocol's own H3b rank-retention rule is inapplicable here for the same
reason: it assumes a rank ratio near 1 and penalizes any drop.

One further caution against over-reading single rungs: seed 902's
`exact_inference` quotient RMS ran 0.011, 0.010, 0.028, 0.055, 0.037, 0.101,
**0.153**, 0.065 — a single excursion past the 0.15 limit at step 1750 that
recovered by 2000. Read at rung 1750 alone it looks like systematic degradation
of the inference corner. It is not.

## 7. Decision

**Switch the campaign to `smooth_100_d001_fp32`, and certify it with a single
50k trained-model audit on the production GPU before spending anything else.**

The evidence for the switch:

- It wins every rung at which its training was healthy, across two seeds, and is
  the only candidate that ever sweeps all seven strata — it does so three times
  (seed 901 at rungs 500 and 1500, seed 902 at the final rung).
- `large_coefficient` failed at **every** rung for `local_1000` and passed at
  **every** rung for `smooth_100` on seed 902.
- On `interior_high`, the stratum that returned the production `NO_GO`, it holds
  9/9 at all twelve measured rungs across both seeds.
- The mechanism was predicted a priori from the fp32 truncation/roundoff
  crossover and then confirmed by measurement, including the 28× versus ~2×
  divergence in how fast each model's time-derivative sharpens. Agreement
  between an independent prediction and the data is most of the reason to act.

`local_1000_d0002_fp32` is not a live option regardless: it has a measured
650k failure that the floor fix provably does not touch.

### What to buy next, in order

The candidate audit needs only a checkpoint and CIFAR-10 — **not** the
regenerated evidence — so the cheapest decisive test runs first:

| step | time | cost @ $0.74/h |
| --- | --- | --- |
| bootstrap + repo update | ~15 min | $0.19 |
| train scale 100 to 50k, production 5,000-update warmup | ~3.0 h | $2.22 |
| `numerical_admission` on it, hardware-bound | ~15 min | $0.19 |
| **decision point** | **~3.5 h** | **~$2.60** |

This single purchase settles both open questions at once: it is the real
numerical certification on the production GPU, *and* it is a stability test at
25× the diagnostic's horizon under the production warmup schedule — which is
the specific condition under which the one observed collapse cannot be
distinguished from a warmup artifact.

Only if it passes do we spend on evidence regeneration (~2.5 h, $1.85) and the
750k foundation (~45 h, ~$33).

### The budget will not stretch to the original plan

The foundation alone is ~$33 and roughly $15 is already spent. The full
campaign as designed — foundation plus the gated ASFD continuation, whose
reserve is $25 — does not fit a ~$35 balance. That decision does not need making
now, because the $2.60 certification is affordable either way and is the correct
next spend. But it should be made with real numbers before the foundation
starts, not discovered at 600k updates.

## 8b. What this does not establish

Conditioning is not quality. A flatter function is easier to differentiate
numerically, and part of `smooth_100`'s advantage may be that a 100× embedding
simply learns a weaker time-dependence — underfitting rather than a win. Nothing
measured here separates those two readings, and CAP-EMF-1 already demonstrated
that passing health gates is compatible with FID 112.94. Only FID on a trained
foundation settles it, which is precisely what the campaign exists to measure.

## 8. Limits

The diagnostic runs at ~2,000 updates against a production horizon of 750,000,
on a different GPU, at a smaller batch. It cannot certify anything, and the
200-update inversion above is direct evidence that short-horizon conditioning
does not simply extrapolate. What it can do is compare two candidates under
identical conditions and show the direction of travel. A candidate that is
already losing ground at 2,000 updates is not going to recover by 750,000.
