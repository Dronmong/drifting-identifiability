# Encoder-Independent Kernel Drifting — Phase 30 results

## Capacity and batch do not unlock coverage. F0 fails its frozen gate — and the gate itself turns out to rest on an unmeasured quantity.

*Protocol: `EncoderIndependentPhase30Protocol.md` (frozen). Pre-flight:
`phase30_preflight.json`. Runner: `run_phase30.py`. Artifact: `phase30.json`
(sha256 `7c5f77de924b4873…`), `phase30.stdout.txt`, 4 sample grids.
4 arms × 2 seeds × 30 000 steps, cloud 256, cosine LR, R11, ~6.3 h.*

---

## 1. Results

| arm | params | positives | KID | precision | **recall** | alpha | ESS | h/seed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `w64_p64` | 146,691 | 64 | **+0.13470** | 0.604 | 0.004 | 4.552 | 0.551 | 0.30 |
| `w64_p256` | 146,691 | 256 | +0.13554 | 0.561 | 0.000 | 4.596 | 0.577 | 0.30 |
| `w128_p256` | 514,563 | 256 | +0.15103 | 0.408 | **0.022** | 4.651 | 0.588 | 0.95 |
| `w192_p256` | 1,103,619 | 256 | +0.13873 | 0.455 | 0.0005 | 4.662 | 0.575 | 1.64 |
| *fresh-latent objectives (6, Ph. 29)* | | | | | *0.000* | | | |
| *memorization ceiling (Ph. 28)* | | | *0.061* | *0.791* | *0.224* | | | |

Per-seed recall: `w64_p64` 0.009 / 0.000 · `w64_p256` 0.000 / 0.000 ·
`w128_p256` **0.044** / 0.000 · `w192_p256` 0.001 / 0.000.

**Verdict: F0 fail.** No arm exceeds the pre-registered recall gate of 0.05.
The best single run is `w128_p256` seed0 at **0.0439** — below the gate, and
contradicted by its own paired seed at 0.000.

Neither metric is monotone in capacity:

- recall ladder (w64 → w128 → w192 at p256): 0.000 → 0.022 → 0.0005
- KID ladder: 0.1355 → 0.1510 → 0.1387

The batch lever is null: −0.0044 on recall, +0.0008 on KID.

---

## 2. The one coherent signal, and it is small

`w128_p256` is the outlier on three measures at once, consistently:
**lowest precision (0.408), highest recall (0.022), worst KID (0.151)**. That is
the shape of a precision/recall trade — slightly more coverage bought with
typicality — and it is exactly what the Phase 23 precision/recall pair was built
to make visible.

It is also driven by one seed of two, so it is an observation and not a finding.

**A methodological note that matters more than the trade.** During this run I
reported three single-seed readings as meaningful and all three failed at the arm
level:

| read live | arm median |
|---|---|
| recall 0.009 (`w64_p64` s0) | 0.004 |
| recall 0.044 (`w128_p256` s0) — called "the most encouraging number this program has produced" | 0.022, gate not met |
| KID 0.117 (`w192_p256` s0) — called "the best fresh-latent KID recorded" | 0.1387, worse than the baseline arm |

Every one regressed. This program's own standing rule (≥8 seeds for close
quantitative readings, in memory since Phase 19) exists for precisely this, and
live monitoring is where it is easiest to ignore. **Single-seed values from a
running job should be reported as telemetry, not as results.**

---

## 3. The gate rests on a quantity nobody measured

The 0.05 threshold was calibrated against measured anchors — 0.000 across six
objectives, 0.224 memorization, 0.496 autoencoder, 0.737 real — which was the
right procedure and a genuine improvement over this program's four
intuition-set thresholds.

**But no anchor establishes the estimator's standard error near zero.** Recall
here is computed from 512 generated samples against 2 048 real, and its
behaviour at low values has never been characterized. Two readings of this
result are therefore both consistent with the data:

- **noise floor ≈ 0.02–0.04** → every nonzero value in §1 is estimator noise,
  the arms are indistinguishable, and capacity has no effect at all;
- **noise floor ≈ 0.005** → `w128`'s 0.044 is real signal that seed1 missed,
  there *is* a capacity effect below the gate, and 2 seeds cannot resolve it.

**This run cannot distinguish them.** I justified 2 seeds on the grounds that
the question was categorical — which holds only if 0.000 is a hard floor rather
than a noisy estimate near zero, and I did not verify that before committing
6.3 h. The pre-flight validated the instrument at the *extremes* (real 0.737,
Gaussian 0.000) and I read that as sufficient; it was not.

**Required before this ladder is interpreted, and before the F1 gate is used:**
bootstrap the recall estimator on a known-zero state at n = 512 and report its
standard error. Minutes of compute. It belongs in the F1 pre-flight regardless
of what is concluded here.

---

## 4. What this settles for the plan

Against §19.4's declared branches this is **F0 fail**: capacity and
target-batch size, over a 7.5× parameter range, do not unlock coverage.
The plan's next step is unchanged and now has its motivating result:

> **F1 — can the encoder-free particle teacher reach nontrivial coverage from a
> distribution the deployed generator actually produces?**

Phase 30 also sharpens two inputs to F1:

1. **The generator is not the binding constraint in this range.** Phase 28
   showed the width-64 net can hold recall 0.224 by memorizing; Phase 30 shows
   7.5× that capacity still yields ~0 under the drifting objective. The deficit
   is not expressiveness.
2. **KID is now demonstrably unusable as the sole readout.** `w128` has the
   worst KID and the best recall; `w192` seed0 had the best KID in the program
   and recall 0.000. Any F1 or F3 gate keyed on KID would have pointed the wrong
   way at least twice in this run.

The `w192` arm also closes the resource question the pre-flight opened: at
cloud 256 the largest configuration that runs without host-memory spill is
width 192 (1 833 MiB peak, 1.64 h per seed). Width 256 at cloud 256 remains
unreachable on this card.

---

## 5. Scope

- 2 seeds per arm — a declared exception justified for a categorical question,
  and §3 is the argument that the justification was not fully earned. No
  quantitative ranking between these arms is licensed.
- Capacity spans 7.5×, not the ~100× separating this harness from a realistic
  generator. This bounds the tested range only.
- Cloud size is fixed at 256 and has never been varied in this program;
  `S2_cloud` died with `S2_wide` in Phase 16 and was never re-run.
- Budget fixed at 30 000 steps. The earlier speculation that duration was a
  hidden third axis (from a 0.009 flicker) is withdrawn — see §2.
- Recall, precision and KID at 512 generated / 2 048 real; comparable only
  between arms measured identically, never with published FIDs.
- Still not the paper's method: pixel-space drift with feature-space kernel
  weights.
