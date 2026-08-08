# CAP-EMF-1 retrospective — the whole procedure, honestly

**Date:** 2026-08-06
**Scope:** design, audit, build, benchmark, run, evaluation — 42.5 GPU-hours, ~$32
**Companion:** [`EncoderIndependentCAPEMF1Results.md`](EncoderIndependentCAPEMF1Results.md)

This is written to be useful to the next run, not to be comfortable. The
science is in the results document; this is about how the work was done.

---

## 1. Was the experiment worth doing?

**Yes**, and I want to say that before the criticism, because the criticism is
long and could mislead.

For ~$32 the run answered a question that four months of prior phases could
not: **encoder-free one-call generation is possible.** Phases 17–30 and B3 had
only ever shown drifting failing at recall 0.0000, and it was genuinely unclear
whether the encoder-free constraint or the tiny generators were responsible.
Now there is a raw-pixel, one-call, encoder-free model producing recognizable
cars, horses and aircraft. That is a real positive in a program built almost
entirely of negatives.

It also produced a **diagnosable failure** rather than a mysterious one, with a
specific, cheap, testable fix. And it prevented a worse decision: forking ASFD
onto a trunk that turns out to be diverged.

The right counterfactual is not "a perfect run" but "going straight to ASFD",
which would have spent the same money building a correction on a foundation
nobody had inspected.

---

## 2. What went well

### 2.1 The pre-run audit paid for itself several times over

Six defects found in code that had already passed 80 tests. Two would have cost
GPU budget directly:

- **The G7 band probes were not band-limited.** `_inverse_haar` applied the
  analysis matrix instead of its transpose; round-trip error 4.31 instead of
  5e-7, and 71–74% of each band's energy landing outside the band it was meant
  to isolate. G7 decides whether the ASFD arm proceeds — and it would have
  **passed**, because four roughly-equal wrong numbers sit comfortably inside
  the declared interval. A silent pass licenses the arm.
- **`calibrate_normalization` allocated a 12.4 GB tensor** at production shape.
  It would have OOM'd on first contact with real data.

Neither was findable by running the test suite. Both were found by reading the
code against what it claimed to do.

### 2.2 The explicit dependency manifest worked exactly as designed

The B2.5 blocker — a glob-based source manifest invalidated by two unrelated
test files, with the recorded bytes unrecoverable — cost that entire
continuation. `stage_cap` and `stage_asfd` hash explicit lists instead. **Mid-run
I added `finalize.py` to `stage_cap/` and the 17-file manifest did not change**,
so the running preflight stayed valid. Under the old design that would have
broken the run.

### 2.3 Recovery, watchdog, and budget-stop all worked

- Recovery resumed at step 74 000 **bit-identically** after a deliberate stop.
- The watchdog detected the 650 000 checkpoint, confirmed the file had stopped
  growing, waited two minutes, stopped the run cleanly.
- The budget-stop rule fixed the result at the last completed checkpoint —
  **and cost us the prettier number**, since 500 000 looks visibly better. That
  it was followed anyway is the point of writing rules down in advance.

### 2.4 The sealed evaluation held

Opened once, after the checkpoint was frozen and hashed, behind an explicit
flag, with the non-primary-checkpoint warning firing as designed. FID against
train and test agreeing to within 1.0 is a bonus: no overfitting to the split.

### 2.5 Looking at pictures repeatedly corrected me

Three times I reasoned from scalars and three times images overruled me:

| I said, from scalars | the images said |
|---|---|
| "residual noise; the sample carries input noise" (250 k) | clear objects, scene layout — **structured, not noise** |
| "HH declining 0.15 per 50 k → 1.0 by 650 k" | HH *rose* by 1.14 over 300 k→500 k |
| implicitly, that later is better | **500 k is visibly better than 650 k** |

The protocol permits train-only visual checks. I should have used them from
step 50 000, not step 250 000.

### 2.6 The time-bucket diagnostic found the root cause

Added in the last pre-run change, specifically because I suspected the sampled
configuration was undertrained and decided to *measure* rather than change an
audited objective. It returned the single most useful number in the run:
**`t ∈ [0.95, 1]` share = 0.000**, at the exact `t = 1` the sampler uses.

Choosing a diagnostic over a speculative fix was the right call and is the one
process decision I would repeat without modification.

---

## 3. What went wrong, ordered by what it cost

### 3.1 The benchmark did not measure the training loop — the root process failure

§8.2 of my own protocol says *"benchmark the exact code"*. The probe I wrote
measured `emf_loss + backward + step` on **synthetic tensors generated directly
on the GPU**. The training loop additionally does, every update: a scattered
gather from a 614 MB CPU tensor, CPU-side RNG with seeded generators, two
host-to-device copies, a horizontal flip, and an EMA update touching 37.7 M
parameters across ~400 kernel launches.

**Measured 0.173 s/update. Reality 0.2289. A 32% error.**

Everything downstream inherited it:

| | projected | actual |
|---|---:|---:|
| foundation | 36.0 h, $27 | **41.3 h, $32** |
| foundation + ASFD fork | ~$34 | unaffordable |
| horizon reached | 750 000 | **650 000** |

The fork was budgeted and then lost. The horizon was cut by 13%. And three
diagnostic detours (§3.2) were spent chasing the discrepancy.

**Fix: benchmark by running the real training loop for 300 updates and reading
its own wall-clock.** The machinery to do that already exists — `train_cap_unit`
with a tiny `updates` value would have been both simpler and correct.

### 3.2 I diagnosed the slowdown wrong twice and deployed a fix on an unverified hypothesis

| attempt | hypothesis | verdict |
|---|---|---|
| 1 | recovery writes every 1 000 updates | **wrong** — rate unchanged after the fix |
| 2 | deterministic algorithms | **wrong** — measured 1.07×, not 1.32× |
| 3 | per-update loop overhead | correct, via timing forensics on recorded history |

The first hypothesis was *plausible* — a 576 MB write every 173 s of compute is
a real cost — and I acted on it without measuring. That required stopping the
run, which is what surfaced the EMA device bug (§3.3), which cost another
restart.

The third attempt used data the run had been recording all along: per-500-step
`wall_seconds` in the history. Bucketing deltas by whether the block ended on a
health step or a snapshot step showed **0.2290 / 0.2286 / 0.2286 — perfectly
uniform**, ruling out every periodic cause in one measurement.

**The lesson is not "I was wrong" but "I had the data to be right on attempt
one and didn't look."** Measure, then fix. A plausible mechanism plus a real
cost is not evidence that the cost is the mechanism.

### 3.3 Three tests that passed for the wrong reason

This is the most important pattern in the whole project.

| test | why it could not fail | found |
|---|---|---|
| EMF derivative vs float64 JVP | a fresh model is **the zero function** (zero-init pixel head, refiner and AdaLN modulation), so the difference quotient and the exact JVP were both identically zero | pre-run audit — only because I asserted *monotone decrease* and `0 > 0` is false |
| restart determinism | runs **CPU-only**, so the EMA shadow and the model are both on the CPU and the device mismatch cannot arise | mid-run, on the first real GPU resume |
| H7 clip fraction | `finalize.py` substitutes `0.0` for a counter the recovery file does not carry, and `0.0 < 0.05` | writing the results document |

Three instances in one project. **A test that cannot fail is worse than no
test**, because it produces confidence proportional to its coverage and delivers
none of it. The derivative test would have certified an unchecked objective if I
had written it the obvious way — asserting the error is small rather than that
it converges at first order.

**Fix: for every test, ask "what state would make this pass vacuously?" and
assert against that state directly.** The repaired derivative test now pins the
zero-function property in its own test so the trap cannot silently return.

### 3.4 The capability gate cannot fail in the direction the run failed

All eight thresholds are floors. H4 requires HH ≥ 0.50 because the predecessor's
failure was too *little* detail (S3R reached 0.159). H3's rank rule caps `best`
at 1, so over-dispersion is invisible by construction.

Result: `PASS, failed=[]` on a model with **6.4× excess diagonal high-frequency
energy and 8.4× excess effective rank**.

I identified this before the run and demonstrated it in rehearsal at step
450 000, then chose not to fix it because any source change invalidates the
hash-bound preflight. That was defensible mid-flight, but it means the gate
contributed nothing to the experiment except a misleading word.

### 3.5 The resume guard vetoed the resume that recovery exists to enable

`run_unit` refused to start whenever a planned checkpoint existed. So the
recovery file could restore the run and the guard would then stop it —
**structurally the same failure that cost the B2.5 continuation, reproduced in
the code written to avoid it.** It fired on the first real interruption.

Writing a lesson down is not the same as internalising it. The manifest lesson
from B2.5 *was* internalised (§2.2); the restart-blocking lesson was not.

### 3.6 The known objective weakness was diagnosed and shipped anyway

I flagged the `t ≈ 1` undertraining before launch, chose to ship a diagnostic
rather than change an audited objective, and that was probably the right call
under time pressure — but it means the run executed a configuration I already
suspected was flawed, and the flaw is the leading explanation for the failure.

A cheap intervention existed: raising the `r` floor from 0.01 to ~0.05 would
have cut the worst coefficient from ≈ 35 to ≈ 7 without touching the time law.

### 3.7 Budget and provider analysis was built on a rate that did not exist

I recommended Community Cloud at ~$0.34/h and built a budget on it. The user's
account only offered $0.74/h. Several exchanges of provider comparison —
including a full ThunderCompute evaluation — rested on an assumed price. I
should have asked what rates were actually visible before analysing.

### 3.8 I lost the material behind the best secondary result

**The 26 post-hoc EMA snapshots (3.9 GB) were never downloaded.** The pod is
gone. So the 26% FID improvement (112.9 → 83.7) is recorded as a number and a
grid image, but **cannot be reproduced, extended, or tuned** — and the averaged
weights themselves were never saved either.

Intermediate checkpoints are likewise gone, so "500 000 looks better than
650 000" can no longer be re-examined beyond the preview PNGs.

I retrieved the *results* and not the *raw material for the finding*. Snapshots
should have been synced to local disk as they were written, at ~150 MB each.

### 3.9 Chained tmux automation was fragile

The `postrun` chain died with the tmux server when the watchdog killed the last
training session. The monitor caught it in ~2 minutes so it cost almost nothing,
but only because I had made the alert fire on a *vanished session* and not just
on a completion marker.

---

## 4. Concrete changes for the next trial

Ordered by expected value.

> **Superseded design note.** The later extended audit rejected bundling a
> higher sampled-`r` floor with endpoint changes: it moves training away from
> the `r=0` inference boundary and does not isolate the sampler. The table is
> retained as the original retrospective, while CAP2 follows the corrected
> matched three-arm design in `EncoderIndependentCAPEMF2ScreenProtocol.md`.

| # | change | why | cost |
|---|---|---|---|
| 1 | **Benchmark the real training loop** — 300 updates of `train_cap_unit`, read its own wall clock | the 32% error that cascaded through everything | trivial |
| 2 | **Raise the EMF `r` floor** from 0.01 to ~0.05 | cuts the worst Equation-18 coefficient from ≈35 to ≈7; a fifth of every batch is currently ill-conditioned | one constant |
| 3 | **Audit inference-corner coverage** before any boundary mixture | the exact `(t,h)=(1,1)` point is unsampled and its joint neighborhood is sparse, although the endpoint bucket is not empty | small diagnostic first; any objective change needs its own audit |
| 4 | **Ceilings in the capability gate** — rank ratio ≤ ~2, HH ratio ≤ ~2 | the gate cannot currently fail on divergence | trivial |
| 5 | **Automatic sample grid every 50 k**, written beside the checkpoint | scalars misled me three times; images corrected me three times | trivial |
| 6 | **Sync snapshots and checkpoints to local disk as written** | §3.8 — the best secondary result is now unreproducible | bandwidth only |
| 7 | **Run the restart test on GPU** in the preflight | §3.3 — the CPU-only test could not see the bug it existed to catch | small |
| 8 | **Treat EMA horizon as a hyperparameter** | post-hoc averaging bought 26% FID for free | measurement only |
| 9 | **Budget from measured rates with 1.5× contingency** | this run needed 1.9× its projection | policy |
| 10 | **Carry windowed counters in the recovery payload** | H7 currently passes on a fabricated `0.0` | small |

Items 1, 2, 4, 5, 6, 10 are all cheap and independent. Items 2 and 3 are the
scientific ones and 3 needs its own float64 derivative audit.

---

## 5. Meta-lessons about how I worked

**Measure before fixing.** §3.2 cost a stop, a restart, and a new bug, all
because I acted on a plausible hypothesis. The decisive measurement took one
script against data already on disk.

**Ask what would make a test pass vacuously.** Three instances (§3.3). This is
now the first question I should ask of any assertion I write.

**Look at the artifact, not the summary statistic.** Five scalars could not
distinguish "emerging images" from "noise with correct amplitude". One 1920×960
PNG settled it instantly, and settled it *against* my stated position twice.

**Writing a lesson down does not internalise it.** B2.5's manifest lesson was
implemented correctly and demonstrably worked. B2.5's restart-blocking lesson
was written in the same audit and then reproduced verbatim in new code. The
difference: the manifest lesson had a *test*; the restart lesson had a
paragraph.

**Predicting a flaw is not the same as controlling for it.** I predicted the
gate would pass regardless, demonstrated it in rehearsal, reported it — and it
still appeared as `PASS` in the final artifact where a reader could take it at
face value. Prediction without remediation is just a well-documented hole.

---

## 6. Open questions the next run should answer

1. **Is the `t ≈ 1` undertraining actually causal for the divergence?** Testable
   cheaply: two short runs differing only in boundary fraction, comparing HH
   trajectories at 200 k.
2. **Does the `r` floor alone fix it?** Independent of (1) and even cheaper.
3. **Where is the true visual optimum?** The trajectory suggests somewhere near
   300 k–500 k. If a fixed run peaks and then degrades, that is a property of
   the objective worth characterising rather than a budget accident.
4. **How much of the 26% post-hoc EMA gain survives a non-diverged run?** If the
   gain is mostly suppression of divergence, it should shrink once (2) and (3)
   land — which would itself be evidence for the diagnosis.
5. **What FID does this architecture reach at DDPM's budget** (102 M images vs
   our 41.6 M)? The gap to published numbers conflates objective, budget and
   one-call sampling, and only the budget axis is cheap to vary.

---

## 7. Standing assessment

The foundation **works and is not good enough to build on**. ASFD's precondition
P1 passes on the letter of its thresholds and fails on the substance: the trunk
whose features ASFD would freeze produces 6.4× excess high-frequency energy, and
ASFD's own G7 gate demands per-band sensitivity within `[0.25, 4.0]` in every
band.

**Recommendation: fix items 1–5, re-run the foundation, and re-assess before
spending anything on the correction.** The cost is comparable to this run and it
addresses a diagnosed cause rather than compounding one.
