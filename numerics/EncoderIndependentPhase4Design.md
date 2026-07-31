# Encoder-Independent Kernel Drifting — Phase 4 design investigation

*Measurement-grounded design of the next phase. Code:
`numerics/encoder_independent_drifting/diagnose_phase4.py`. Artifacts:
`phase4_design.json` (G1), `phase4_g2.json` (G2). Development seeds
(`MASTER_SEED + 2000..`), disjoint from the Phase-3 confirmation seeds.
Nothing here feeds a gate.*

---

## 0. Executive summary

Two questions decided the next phase, and I measured both rather than
arguing them:

| # | Question | Answer |
|---|---|---|
| **G1** | Is the anchor's benefit just an indirect variance fix? | **No.** It survives R11 unchanged (0.962 with, 0.966 without; 3/3 wins each). The two effects are orthogonal — **but on raw geometry the anchor is worth only ~3.5%** |
| **G2** | Is the contraction R11 fixes specific to my setup? | **No, it is universal.** Effective dimension collapses to **0.218–0.308** in every configuration tested — batch 32/64/128, raw and wavelet geometry — and R11 recovers 0.25–0.33× in all of them |

These point in one direction. **The anchor is too small to carry a phase.
R11 is the finding, and it is not about encoder-independence at all** — it is
a property of stop-gradient regression onto a mean-shift teacher, which is
the paper's own training recipe. Phase 4 should establish that.

---

## 1. G1 — the anchor confound is refuted, and the anchor is small

Every anchor measurement this program ever made was taken on a generator now
known to be collapsed to ~2.3 effective dimensions. Since the anchor pulls
output back toward the data manifold, its benefit might have been nothing
but an indirect variance correction. A 2×2 factorial settles it:

| arm | score | ED² | effective dim | coverage |
|---|---:|---:|---:|---:|
| plain | 7.578 | 2.292 | 0.254 | 0.951 |
| anchor | 7.270 | 2.206 | 0.262 | 0.953 |
| R11 | 1.999 | 0.223 | 1.001 | 0.934 |
| anchor + R11 | **1.880** | 0.212 | 0.998 | 0.932 |
| *skyline* | *1.628* | *0.153* | *0.874* | *0.953* |

Paired ratios:

| comparison | ratio | interval | wins |
|---|---:|---|---:|
| anchor, **without** R11 | 0.9662 | [0.9593, 0.9765] | 3/3 |
| anchor, **with** R11 | 0.9621 | [0.9404, 0.9881] | 3/3 |
| R11, **without** anchor | 0.2517 | [0.2184, 0.2767] | 3/3 |
| R11, **with** anchor | 0.2506 | [0.2241, 0.2716] | 3/3 |

**The confound hypothesis is refuted.** The anchor's benefit is identical
with and without R11, and R11's is identical with and without the anchor.
They are orthogonal, additive, and both replicate 3/3. The anchor is active,
not decorative: its gradient share is 0.146–0.175, well above the frozen 0.05
presence threshold.

**But the magnitude matters for planning.** On raw geometry the anchor is
worth **3.5%**. The 30–45% figures I have been quoting (A5/A4 = .701,
A6/A4 = .658, B3/B1 = .906) were all measured on *structured-geometry* arms —
wavelet and random-convolutional — which the program has abandoned. That is
consistent with what the anchor is for: it compensates for a geometry that
is not measure-determining, and the raw pixel kernel largely is. On the
geometry we now use, there is much less for it to fix.

A 3.5% effect is confirmable but is not worth a phase of its own. It should
ride along as a secondary arm.

---

## 2. G2 — the contraction is universal

R11 was derived on raw geometry at batch 64. If the contraction is a general
property of the training recipe rather than an artifact of that corner, it
must appear elsewhere.

Effective dimension **without** R11, all nine runs:

```
0.218  0.222  0.227  0.228  0.232  0.232  0.296  0.301  0.308
```

Every configuration — batch 32, 64 and 128, raw and wavelet geometry —
collapses to between a fifth and a third of the data's effective dimension.
R11's paired effect, within seeds:

| configuration | R11 ratio | interval |
|---|---:|---|
| raw, batch 32 | 0.3265 | [0.2582, 0.4693] |
| raw, batch 64 | 0.2517 | [0.2184, 0.2767] |
| raw, batch 128 | 0.2659 | [0.2227, 0.3684] |
| wavelet, batch 64 | 0.2780 | [0.1955, 0.4198] |

A 3–4× improvement everywhere. **This is not a property of raw pixel
geometry, of a particular batch size, or of encoder-independence. It is a
property of regressing a generator onto a mean-shift teacher.**

That is also the paper's training recipe, which makes it the most
transferable result this program has produced — and the reason Phase 4
should be about R11 rather than about anchors or geometry.

### The geometry verdict is unaffected

| comparison | ratio | interval | wins |
|---|---:|---|---:|
| wavelet / raw, **without** R11 | 1.231 | [1.211, 1.262] | 0/3 |
| wavelet / raw, **with** R11 | 1.360 | [0.891, 2.327] | 1/3 |

Raw still wins under the repair; the point estimate moves against wavelet,
though the interval widens. Nothing here reopens the geometry thread.

---

## 3. What is left after R11

| quantity | value |
|---|---:|
| corrected baseline (raw + R11) | 1.999 |
| with anchor | 1.880 |
| sliced-Wasserstein skyline | 1.628 |
| metric floor (fresh real sample) | ≈1.2–1.7 |

The corrected baseline is roughly **1.15–1.23× the skyline** and still short
of the floor. Phase 3 called that parity because the confidence interval
touched 1.0; at these development seeds the skyline is still slightly ahead.
So there is real residual headroom, and it is currently unexplained — that is
worth one diagnostic, not a research programme.

---

## 4. Phase 4: establish R11 as a general property

### 4.1 The claim to test

> Stop-gradient regression onto a mean-shift teacher is
> **variance-contracting**, the contraction compounds over training, and
> matching the teacher's second moment to the data's removes it — across
> batch sizes, resolutions, kernel geometries, and at the paper's own
> declared operating point.

This is a claim about a *training recipe*, not about encoder independence.
It is falsifiable, cheap to test, and if it holds it is the first thing in
this program that would transfer outside it.

### 4.2 P4A — generality confirmation *(fresh seeds, frozen)*

A grid crossing the axes that could plausibly matter, with R11 on/off:

| axis | values |
|---|---|
| batch | 32, 64, 128 |
| resolution | 16, 32 |
| geometry | raw, wavelet |
| **temperature** | the paper's declared grid, τ ∈ {0.02, 0.05, 0.2} |
| **self-mask** | off, **on** (Algorithm 2's `eye(N)*1e6`) |

The last two axes have **never been tested in this program** and are the
paper's actual operating point (`numerics/README.md`, Table 8 / A.6). If the
contraction survives them, the claim is about the paper's recipe rather than
about my parameter choices.

Gate: R11's paired ratio ≤ 0.60 in **every** cell, with effective-dimension
ratio rising above 0.60 in every cell, on fresh seeds.

### 4.3 P4B — cross-harness validation *(the strongest single check)*

Reproduce the contraction in **`numerics/lowdim_drift.py`** — this
repository's independently audited, verbatim Algorithm-2 harness, which is
cross-checked against `driftlab.compute_v_paper` and has its own Lean
crosswalk.

If a neural generator trained by stop-gradient regression in *that* harness
also collapses, and the same second-moment correction repairs it, then the
finding is not an artifact of my package. This is cheap (1-D and 2-D
targets) and is the single most convincing thing Phase 4 could produce.

Gate: the contraction appears (effective-dimension ratio < 0.5 without the
fix) and is repaired (> 0.8 with it) in a harness this program did not write.

### 4.4 P4C — analytic characterization

The contraction should be predictable, not merely observed. For a Gaussian
kernel the mean-shift teacher is a known smoothing operator; the per-step
variance ratio of `x + ηV` should be computable, and the fixed point of the
compounded map should predict the ≈0.22–0.31 effective-dimension ratio
measured. Deriving that — even for a tractable special case — would convert
an empirical repair into a mechanism, which is this repository's usual
standard and would make the result worth formalizing.

Gate: a closed-form per-step contraction factor that predicts the measured
ratio within a stated tolerance on at least one tractable target family.

### 4.5 P4D — residual-gap diagnostic *(scoped, one experiment)*

After R11 the baseline is 1.15–1.23× the skyline. One diagnostic to localize
it: compare per-component ratios and the effective-dimension *spectrum*
(not just its participation ratio) between the corrected generator, the
skyline generator, and free particles. Either it identifies a second
mechanism or it shows the gap is estimator noise; both are useful and it
should not become a programme.

### 4.6 The anchor rides along

Include anchor on/off as a secondary factor in P4A rather than giving it a
phase. Its effect on raw geometry is 3.5%, replicated 3/3 and orthogonal to
R11, and it is the program's only source-correctness mechanism. Report it;
do not build around it.

---

## 5. What Phase 4 is *not*

- **Not a geometry screen.** Three negatives, plus a re-check under R11
  (1.360, raw still wins). Closed.
- **Not an anchor confirmation.** 3.5% on the geometry we use; it rides along
  in P4A instead.
- **Not scaling.** Establishing the mechanism at 16–32 px is worth more than
  another resolution, and the analytic component (P4C) is where the leverage
  is.
- **Not a claim about the paper's results.** Even if P4A and P4B pass, the
  finding is that *this training recipe* contracts and can be corrected. The
  paper trains with a feature encoder at a scale and batch this does not
  touch, and may already avoid the problem by other means.

---

## 6. Honest assessment of where the program stands

Four phases have produced one confirmed positive (R11) and three solid
negatives (fixed compositional geometry, twice on synthetic data and once on
real). The positive is an implementation repair, not a new method — it takes
encoder-free drifting from *clearly worse* than a trivial baseline to
*roughly level* with it.

The strongest remaining move is to establish that the repair is general and
mechanistic (P4A–P4C). If it is, the transferable claim is a caution about a
widely-used training pattern, evidenced in two independent harnesses with an
analytic account. If P4B fails — if the contraction does not appear in
`lowdim_drift` — then R11 is a local fix to my package, its scope shrinks
accordingly, and that should be reported plainly.

## 7. What this investigation does not establish

- G1 and G2 are development measurements on three seeds at one resolution;
  the P4A grid is what would confirm them.
- The anchor's 3.5% is measured on raw geometry only. Its larger historical
  effects on structured geometry were real, but on arms the program no
  longer runs.
- The residual skyline gap is unexplained, and P4D may not explain it.
- Nothing here touches ImageNet, FID, or the paper's trained model.
