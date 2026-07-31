# Encoder-Independent Kernel Drifting — Phase 2 failure diagnosis

*Post-hoc investigation of `EncoderIndependentPhase2Results.md`. Code:
`numerics/encoder_independent_drifting/diagnose_phase2.py`. Nothing here
feeds a gate. Two claims in the Phase-2 results are overturned below and one
is confirmed.*

---

## 0. Executive summary

Phase 2 reported two things: fixed geometry is 37% worse than raw pixels, and
**every** drifting arm is ~4× worse than a sliced-Wasserstein skyline. The
second was flagged as "the biggest finding of the phase". It was an
implementation defect.

| # | Question | Answer |
|---|---|---|
| **F1** | Is the missing column reweighting the cause? | **No.** Implementing the paper's real Algorithm 2 helps raw 13% and wavelet 1%; the ranking and the skyline gap survive |
| **F2** | Is the field starved at batch 64? | **No.** Batch 32→256 improves drifting 9.25→5.38 but the skyline ratio stays 3.2–4.1× |
| **F4** | Is the gap in the field or in the generator? | **The generator.** The *same* field on free particles scores **1.89** against the skyline's 1.84 — parity |
| **F5** | Does a stronger (rollout) teacher fix it? | **Barely.** 7.47 → 6.60, ~11% |
| **F6** | What exactly is wrong with the generator's output? | **Variance collapse**: effective dimension **2.34** against the data's 8.32 |
| **F7** | Does correcting it close the gap? | **Yes, entirely.** 6.69/7.05 → **1.92/1.30**, matching or beating the skyline |
| **F8** | Does the geometry verdict survive the fix? | **Yes.** Raw 1.93 still beats wavelet 3.44 by 1.8× |

**The dominant fact of Phase 2 — that drifting is far weaker than a simple
distribution-matching loss — was an artifact of a contraction in the
stop-gradient regression, and one scalar correction removes it.** The
geometry conclusion is unaffected.

---

## 1. A discrepancy found by re-reading the plan against the repository

Every phase of this program has used the field written in plan §6.3 as "the
paper's standard normalized displacement field": the row-normalized SNIS
mean shift. The repository contains a verbatim port of the paper's actual
Algorithm 2 (`lowdim_drift.drift_paper`, cross-checked against
`driftlab.compute_v_paper`) and it is **not** that field — it normalizes the
affinity matrix along *both* axes and weights each side by the other's total
mass. `lowdim_drift` labels the SNIS field "DIAGNOSTIC ONLY", and there is a
Lean development around the omitted term
(`ColumnReweightedMeanShift.lean`) plus a numerics experiment (E2) measuring
its scale.

The two differ by 70% in norm on identical inputs. More usefully, the
difference is *structured*: per-particle direction is nearly identical
(cosine 0.999), but the magnitude is damped by local density —

| region | ‖V_paper‖ / ‖V_snis‖ |
|---|---:|
| dense clump | 0.39 |
| sparse tail | 0.71 |

— and under SNIS a single target point can exert **7.3×** more total pull
than another with no correction. The column normalization divides a target
point's influence by the attention it receives, which is an
anti-density-seeking mechanism, and density-seeking was exactly the failure
Phase 2 measured.

That made it the leading hypothesis. `direction_mode="paper"` is now
implemented, generalized to any positive-definite block kernel (working from
the Gram matrix, since `softmax(logits) = k / Σk`), and **verified to
reproduce `lowdim_drift.drift_paper` to 1e-4**.

### F1 — the hypothesis is wrong

Median over 3 seeds, CIFAR-16, 600 steps:

| arm | SNIS | paper Algorithm 2 |
|---|---:|---:|
| raw | 7.49 | **6.52** |
| wavelet | 10.12 | 10.05 |
| *skyline* | *1.84* | *1.84* |

The correct estimator is worth ~13% on raw and nothing on wavelet. It does
not change the geometry ranking and does not touch the skyline gap. The
discrepancy is real and worth fixing on its own merits — the program has been
testing a field the repository elsewhere calls a diagnostic — but it is not
the explanation.

### F2 — nor is batch size

| batch | raw (paper) | skyline | ratio |
|---:|---:|---:|---:|
| 32 | 9.25 | 2.23 | 4.1× |
| 64 | 6.52 | 1.84 | 3.5× |
| 128 | 5.83 | 1.63 | 3.6× |
| 256 | 5.38 | 1.67 | 3.2× |

Drifting improves with batch, but so does the skyline; the ratio is flat.

---

## 2. F4 — the field is sound; the generator is not

G0.5 certified that the raw field's zero-set is reachable — but it tested a
*free particle cloud*, which can take any configuration. The generator maps a
32-dimensional latent through a smooth network, so its reachable set is a
low-dimensional manifold. Those are different questions, and the entry gate
only answered the first.

Running the **identical field** on both:

| realization | score | ED² |
|---|---:|---:|
| free particles, raw + paper field | **1.89 / 1.92** | 0.155 / 0.160 |
| free particles, raw + SNIS field | 2.19 / 2.06 | 0.184 / 0.188 |
| *skyline (same generator, SW loss)* | *1.84 / 2.20* | *0.172 / 0.263* |
| **generator, raw + paper field** | **6.52 / 8.15** | 1.49 / 2.25 |
| free particles, wavelet | 6.55 / 6.70 | 1.44 / 1.43 |

Two readings:

1. **The drifting field reaches skyline quality.** Free particles under the
   raw field score 1.89 against the skyline's 1.84. There is nothing wrong
   with the field, the kernel, or the bandwidth.
2. **The generator loses a further 3.5×** — and the same generator with a
   sliced-Wasserstein loss does not. So it is neither the field nor the
   generator's capacity: it is the *coupling* between them.

The wavelet field is 3.5× worse than raw **even on free particles**, where
no generator can be blamed. That independently confirms the Phase-2 geometry
verdict.

---

## 3. F6 — the coupling defect is variance collapse

Geometry of each cloud (participation ratio = effective dimension):

| cloud | median pairwise distance | RMS | **effective dim** |
|---|---:|---:|---:|
| CIFAR-16 (real) | 17.62 | 0.489 | **8.32** |
| skyline generator (SW) | 16.95 | 0.459 | 7.27 |
| free particles (drifting field) | 14.90 | 0.420 | 6.15 |
| **drifting generator** | **8.80** | **0.281** | **2.34** |

The drifting generator has collapsed onto a roughly two-dimensional manifold
with half the data's spread. The same field on particles keeps 6.15; the same
generator under SW keeps 7.27.

**The mechanism is a double contraction.** The stop-gradient target
`x + ηV` moves every sample toward its neighbourhood barycentre, so the
teacher cloud is narrower than the data. Least-squares regression onto noisy
targets shrinks variance again — ordinary regression to the mean. Iterated
600 times, the two compound. Free particles escape it because the negative
(repulsion) term acts on the cloud directly instead of through a fit; the
skyline escapes it because its loss compares *sorted* generated and target
projections, which cannot be reduced by shrinking.

This also explains the Phase-2 per-component pattern that was previously
described only as "density-seeking": high precision and `nearest_real`
(samples land in dense regions), poor `ed2`/`sw1`/`patch_ed2` (the
distribution is too narrow).

---

## 4. F7 — one scalar closes the entire gap

**Reform R11.** Rescale the stop-gradient teacher about its own mean so that
it carries the real batch's second moment. It is a single scalar, so no
sample changes direction — the field's decisions about *where* each sample
goes are preserved; only how far the cloud may shrink on the way is changed.

CIFAR-16, 600 steps, raw + paper field:

| seed | rollout | teacher variance match | score | ED² | effective dim |
|---:|---:|---|---:|---:|---:|
| 0 | 1 | off | 6.686 | 1.608 | 1.98 |
| 0 | 1 | **on** | **1.919** | **0.177** | **6.67** |
| 0 | 4 | on | 2.019 | 0.173 | 5.27 |
| 1 | 1 | off | 7.050 | 1.805 | 2.06 |
| 1 | 1 | **on** | **1.298** | **0.121** | **6.73** |
| 1 | 4 | on | 2.096 | 0.227 | 4.84 |

A **3.5–5.4× improvement**, effective dimension restored from ~2.0 to ~6.7,
and the result now **matches or beats the skyline** (1.92 vs 1.84; 1.30 vs
2.20) on both ED² and the composite. The rollout teacher (F5) is unnecessary
once variance is corrected, and at K=4 slightly worse — simpler is better.

R11 is implemented as an opt-in `ObjectiveConfig.teacher_variance_match`
(default off, so the paper-style behaviour is reproduced exactly), with three
unit tests.

---

## 5. F8 — the geometry verdict survives the repair

The obvious worry: if a defect dominated the absolute numbers, did it also
distort the raw-versus-wavelet comparison? Median over 3 seeds:

| geometry | variance match off | variance match on |
|---|---:|---:|
| raw | 7.618 | **1.931** |
| wavelet | 13.220 | **3.442** |

The correction is worth ~3.9× to raw and ~3.8× to wavelet — almost exactly
equal — and **raw still beats wavelet by 1.8×**. With the fix the wavelet
arm's coverage falls to 0.46–0.58 against raw's 0.95, so its samples are
spread but wrong.

Together with F4's free-particle result (wavelet 6.6 vs raw 1.9, no generator
involved), the Phase-2 geometry conclusion stands on two independent legs.

**Caveat on this table.** The quick script used only `branches[0]` of each
family, so its "scattering" rows were the same wavelet branch and are not
reported. A full multi-branch re-run belongs in the reformed screen, not in a
diagnostic.

---

## 6. Corrections to `EncoderIndependentPhase2Results.md`

| claim | status |
|---|---|
| "fixed geometry is 37% worse than raw pixels" | **confirmed**, and independently reconfirmed on free particles and after R11 |
| "the anchor result replicates" | unaffected |
| "drifting itself is the weak link, not the geometry" | **withdrawn.** The gap was a variance contraction in the transfer to the generator, not a property of drifting |
| "no choice of geometry recovers that" | **withdrawn** — the premise is gone |
| "the interesting question may be whether the drifting objective is competitive at all" | **withdrawn.** With R11 it is competitive: 1.92 against the skyline's 1.84 |
| G0.5 (E3) as evidence that the arms' objectives were sound | **narrowed.** It certifies the field on *free particles*; it says nothing about whether a parametric generator can realize that zero |

---

## 7. Reforms

| ID | Reform | Status |
|---|---|---|
| **R11** | teacher variance match | implemented, opt-in, 3 tests |
| **R12** | `direction_mode="paper"` — the real Algorithm 2 | implemented, verified against `lowdim_drift.drift_paper`, 2 tests |
| **R13** | G0.5 must be run *through the generator*, not only on free particles | **specified below, not implemented** |
| **R14** | report the generated cloud's effective dimension every run | **specified below, not implemented** |

### R13 — parametric zero-set reachability

G0.5 as written asks "can this field's zero be reached by *some* cloud".
The question that matters is "can it be reached by *this generator*". The
successor should run both and report the pair; the difference between them is
exactly the coupling defect this diagnosis found, and it would have been
visible before Phase 2 was run.

### R14 — effective dimension as a standing diagnostic

Participation ratio of the generated cloud, against the data's, is a
one-line, cheap statistic that made a 3.5× defect obvious after four phases
of not looking. It belongs in the standard per-run diagnostic set alongside
ESS and coverage.

---

## 8. What this changes about the next phase

The Phase-2 exit gate stands: fixed compositional geometry does not help, and
that is now supported by three independent measurements (generator arms, free
particles, and after R11). **The geometry thread remains closed.**

What re-opens is the *baseline*. With R11, encoder-free raw-pixel drifting on
CIFAR-16 reaches a composite of ~1.3–1.9 against a metric floor of ≈1.2–1.7
and a sliced-Wasserstein skyline of 1.84. That is a substantially stronger
result than anything this program has produced, and it was invisible until the
contraction was removed.

The indicated next step is therefore **not** another geometry screen but a
re-measurement of the corrected baseline: does encoder-free drifting with
R11 + R12 hold up across seeds, resolutions and budgets, and how close does
it get to the metric floor? That is a well-posed question with a positive
expected answer, and the anchor thread (which has now helped in every
configuration it has been placed in) attaches to it directly.

## 9. What this pass does not establish

- R11 is validated on CIFAR-16, raw geometry, 2–3 seeds, 600 steps. It is a
  measured repair of a measured defect, not a confirmed method.
- The F8 comparison used a single branch per family; a full multi-branch
  screen has not been re-run with R11.
- No claim about ImageNet, FID, or the paper's model. The paper trains at a
  scale and with an encoder this does not touch, and it may not exhibit this
  contraction at all — its own protocol differs in batch, schedule, and
  feature space.
- The skyline remains a *distribution-matching* reference at one budget, not
  a claim about generative quality in general.
