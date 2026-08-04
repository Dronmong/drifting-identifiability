# CAP-EMF-1 local port audit

**Date:** 2026-08-04
**Scope:** protocol section 8.1 — everything up to the run, and nothing beyond it
**Status: GO for the cloud benchmark. Not GO for the training run.**

Artifacts: `stage_cap/cap_preflight.json` (sha256 `22139104e6c65266…`), verified
sidecar, `verdict.decision = GO`, 12 hashed sources.

---

## 1. What was built

```
numerics/encoder_independent_drifting/stage_cap/
    config.py       frozen dataclasses, capability + smoke profiles, feature levels
    model.py        patch-2 U-ViT, AdaLN-Zero, local pixel refiner, feature taps
    objective.py    direct-x EMF (JVP-free) + a float64 JVP reference
    diagnostics.py  endpoint health, Haar bands, corrected rank rule, capability gate
    data.py         automobile pools; the test split is sealed by construction
    artifacts.py    explicit dependency manifest, hashing, checkpoint discipline
    training.py     training loop, EMA, checkpoints, mid-unit recovery
    preflight.py    the nine section 8.1 checks
    run_unit.py     the budget run, refused without explicit opt-in
    tests/test_cap.py
```

**26/26 regression tests pass. 9/9 preflight checks pass.**

---

## 2. The nine checks

| # | check | result |
|---|---|---|
| 1 | shape identities | output `(2,3,32,32)`, **256 tokens** |
| 2 | EMF difference vs float64 JVP | rel. error `1.17e-1 → 1.10e-2 → 1.10e-3`, **first-order rates 10.62, 10.02** |
| 3 | one-call inference | **1** model evaluation |
| 4 | restart determinism | final `raw_mse` identical to 12 digits |
| 5 | EMA arithmetic and maturity | half-life 6 931; mature by 34 656; residual initialization weight at 160 k = **1.12e-07** |
| 6 | Haar energy + corrected rank rule | conserved; accepts S3R EMF 1.661; rejects pMF 0.349 |
| 7 | feature-tap parity + 16×16 grid | **bit-identical**; four levels present |
| 8 | parameter count | **63 548 687** = 97.8% of the 65 M ceiling |
| 9 | throughput | micro_batch 16 fits: **2.240 s/update, 4.35 GiB, 99.6 h** on the RTX 4050 |

Check 2 is the one that matters most. The rates of 10.62 and 10.02 for
tenfold reductions in `δ` are textbook first order, which no sign error, clock
error, or wrong Euler velocity survives. The clock substitution
(`1/(1−t_paper)² → 1/t²`, `0.02` clamps on local `t` and `r`) is therefore
confirmed against the derivative it is supposed to approximate, not merely
transcribed.

---

## 3. Three findings from the audit itself

### 3.1 The derivative test was vacuous, and only failed by luck

The first run of check 2 failed. The cause was not the objective:

> **A freshly constructed model is the zero function.** `pixel_head` is
> zero-initialized, the refiner's final convolution is zero-initialized, and
> AdaLN-Zero zeroes every modulation. So the output is identically zero for
> every input, and both the difference quotient and the exact JVP are
> identically zero.

The comparison proved nothing. It failed only because it asserted *monotone*
decrease and `0 > 0` is false. Had it asserted only "the error is small" — the
obvious way to write this test — it would have **passed for the wrong reason**
and certified an objective nobody had checked.

Repaired with `wake_output_path`, which randomizes the deliberately
zero-initialized output path before any derivative audit, and by asserting the
convergence *rate* rather than a magnitude. The zero-function property is now
pinned by its own test so the trap cannot silently return.

### 3.2 The refiner is dead for exactly one update

Traced while checking 3.1, then verified rather than assumed. At
initialization `base = 0`, and every refiner bias is zero, so every intermediate
activation inside the refiner is zero and its final convolution receives **no
gradient on the first update**. It wakes on update two, once `pixel_head` has
moved.

Benign — but a *permanently* dead refiner would be a silent loss of exactly the
capacity added to fix S3R's HH deficit, and it would not show up in any loss
curve. `test_refiner_is_dead_for_exactly_one_step_then_trains` now asserts the
gradient is zero on update one and nonzero on update two.

### 3.3 The first throughput probe was 53% pessimistic

It measured micro_batch 4 and scaled by 16×, including warmup steps, and
reported **152.2 h**. Measuring the production shape gives **99.6 h** on the
same GPU:

| micro_batch | s/microbatch | reserved | s/update | hours |
|---:|---:|---:|---:|---:|
| 4 | 0.1667 | 1.61 GiB | 2.667 | 118.5 |
| 8 | 0.2995 | 2.44 GiB | 2.396 | 106.5 |
| **16 (production)** | **0.5590** | **4.31 GiB** | **2.236** | **99.4** |

A small-microbatch probe is latency-bound and systematically understates
throughput. The check now starts at the production microbatch and steps down
only on OOM, and excludes warmup.

---

## 4. Open gap: the protocol says BF16, the implementation is FP32

Protocol section 3 specifies *"BF16/AMP; sensitive reductions and all
diagnostics in FP32."* **`training.py` uses no autocast at all.** This is the
one place where the build does not match the run card, and it is recorded
rather than quietly resolved.

There is a numerical reason to be careful, not just to switch it on:

> The EMF local difference computes `(future − current) / δ` with `δ = 0.01`,
> where `future` and `current` are two evaluations of the *same network* at
> nearby inputs. It is a difference of nearly-equal quantities divided by a
> small number — the textbook setting for **catastrophic cancellation**. BF16
> carries roughly 2–3 significant decimal digits, so in BF16 the quotient can
> be dominated by rounding noise rather than signal.

So "sensitive reductions in FP32" must name the local difference explicitly. My
recommendation:

1. **Benchmark in FP32 first.** It is what is implemented, verified, and
   hashed, and section 8.2 requires benchmarking the *exact* code.
2. Treat BF16 as a follow-up optimization that must re-pass check 2 — the
   first-order convergence test — before being adopted, with the local
   difference kept in FP32 regardless.

FP32 is affordable (§6), so this is a cost optimization, not a blocker.

---

## 5. Design decisions the port forced

**The conditioning bottleneck was necessary.** Per-block DiT-style
`Linear(width, 6·width)` modulation costs 1.57 M × 12 = 18.9 M parameters and
would have put the model near 72 M — over the ceiling. A shared conditioning
trunk projected to `condition_dim = 256`, then per-block
`Linear(256, 6·512)`, costs 9.4 M and lands at **63 548 687**. The protocol
named this as the first reduction to try; it turned out to be required, not
optional.

**97.8% of the ceiling is tight.** Any width or depth change blows it. That is
the intended behaviour — the ceiling exists so capacity changes are deliberate —
but it means the "reduce width once" escape in the protocol's no-loop rule is
the only remaining headroom.

**`_DEPENDENCIES` is an explicit list of 12 files, and `tests/` is not in it.**
This is the B2.5 lesson implemented rather than described:
`test_source_manifest_is_an_explicit_list_not_a_glob` creates a scratch module
beside the stage and asserts the manifest does **not** change. Under
`b1_freeze.py`'s globbing that same act permanently invalidated a completed
confirmation and cost the entire B2.5 continuation.

**The test split cannot be opened by accident.** `sealed_test_pool` raises
unless called with `acknowledge_sealed=True`, and `training.py` never imports
it. A separate test asserts the guard fires.

**`training.py` can resume mid-unit.** B2.5 could not, which is why an
interrupted unit lost every completed hour and then blocked its own restart.
Optimizer, model, EMA and all four RNG streams are written atomically every
1 000 updates, and `test_restart_reproduces_an_uninterrupted_run` asserts a
resumed run reaches the same loss as an uninterrupted one.

**A test asserts no correction term can leak in.**
`test_no_correction_term_is_importable_from_this_stage` greps the training
module for `laplace`, `sinkhorn`, `spectral_anchor` and `drift_energy`. The
whole point of CAP-EMF-1 is that a failed foundation cannot be mistaken for a
failed correction.

---

## 6. Cost projection

**99.6 h on the RTX 4050 laptop at the production shape, FP32.**

Scaling to a rented 4090 by FP32 throughput (~82 vs ~12 TFLOPS) and memory
bandwidth (1008 vs 192 GB/s) gives a realistic 5–7× band:

| | estimate |
|---|---|
| RTX 4090, FP32 | **14–20 h** |
| at $0.35–0.70/h | **$5–14** |
| with working BF16 (§4) | roughly half again |

Memory is comfortable: 4.35 GiB at micro_batch 16, against 24 GiB on a 4090 —
so the rented run can likely use a larger microbatch with no accumulation,
which the benchmark should test.

**These remain projections.** Section 8.2 exists because this class of estimate
is routinely wrong by 2–3×, and the benchmark selects by measured dollars per
update on the actual instance.

---

## 7. What is deliberately not built

**The sealed evaluation (protocol section 7.2).** Uncurated grids, class
recognizability, KID, report-only FID, precision/recall, density/coverage,
spectrum, and the memorization audit.

It runs after the final checkpoint is frozen, so it does not block the run and
costs no idle GPU time. But there is a preregistration argument for building it
**before** the run rather than after: an evaluator written while looking at
training curves is an evaluator that can be tuned. I would build it next, and
hash it into the preflight, so the sealed evaluation is fixed before any data
exists to tune it against.

**The cloud benchmark harness (section 8.2)** is an operational step on rented
hardware, not local code.

---

## 8. Verdict

**GO for the $1 cloud benchmark. The training run stays blocked** — `run_unit`
refuses without `--i-have-authorized-the-budget-run`, and that opt-in should not
be given until the benchmark has projected the cost on the actual instance.

The gate order from here:

1. build the sealed evaluator and re-hash the preflight (recommended);
2. resolve the BF16 question, or record FP32 as the frozen choice;
3. $1 benchmark on the exact code; select by measured dollars per update;
4. project the full cost and compare against the reserved budget;
5. only then authorize the run.
