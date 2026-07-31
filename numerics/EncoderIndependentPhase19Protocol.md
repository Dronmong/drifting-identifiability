# Encoder-Independent Kernel Drifting — Phase 19 protocol

## Two cheap mechanism fixes, screened before the long-budget run

**Frozen before the run. Declared outcomes in §6. Nothing below is tuned
against a result.**

---

## 1. Why this runs before the scaling run

The scaling ladder (FID 292 → 288 → 259 → 232 over 600 → 30 000 steps) was
still falling when it stopped, and the obvious next move is a 100 000-step
run. That run is cheap in wall time (~2 h for two seeds at raw geometry) but
it would characterize a configuration we already have evidence is
misconfigured in two ways.

### F1 — the bandwidth is set to the value our own audit measured as worse

`run_phase16.py:49` sets `GOOD_ESS = 0.9`, and Phases 16, 17 and 18B all
inherit it. `EncoderIndependentMetricAudit.md` measured this knob directly
against FID:

| | FID | ED² |
|---|---:|---:|
| ESS 0.5 + R11 | **244.0** | 0.3054 |
| ESS 0.9 + R11 | 258.8 | 0.4224 |
| ESS 0.5, no R11 | 260.3 | 1.8985 |
| ESS 0.9, no R11 | 266.8 | 2.2746 |

ESS 0.5 is ~15 FID better in both R11 conditions. The program is at 0.9
because Phase 7 found it 4.9× better on **ED²** — the metric the audit then
showed a structureless Gaussian saturates. The audit flagged this and the
constant was never changed.

**The gap was measured at one short budget.** It may not survive to 15 000
steps, which is exactly why this is screened rather than flipped.

### F2 — the algorithm has no annealing path at all

Three established facts compose:

- the drift is **RMS-normalized** (`FieldConfig.normalization = "rms"`), so
  the teacher sits a *constant* distance from the current output however
  close the cloud is to the data;
- **`step_eta` is inert under Adam** (R24: the stop-gradient gradient is
  `−2ηV/n` and Adam is invariant to constant gradient rescaling);
- the learning rate is a constant `2e-3` — there is no scheduler anywhere in
  the package, and no weight EMA (the only `ema` is the adaptive-mixture
  controller, unrelated).

⟹ **the generator takes constant-magnitude steps forever; no fixed point
exists.** It can only reach a statistical equilibrium and jitter about it —
consistent with the shape law, which predicts an equilibrium *radius* rather
than a limit.

The two standard remedies are weight EMA (universal in generative modelling:
ProGAN/StyleGAN, score-based and diffusion models) and learning-rate decay.
By R24, LR decay is the *only* remaining annealing lever in this recipe.

---

## 2. Design

**EMA does not change training.** It is a second copy of the weights updated
alongside the live ones, so a single training run scores both the live and
the EMA weights from an identical trajectory. That makes EMA an *evaluation*
factor, perfectly paired, and collapses a 2×2×2 design into **four training
arms**:

| arm | target ESS | LR schedule |
|---|---|---|
| `A_ess9_const` | 0.9 | constant (the as-is baseline) |
| `B_ess5_const` | 0.5 | constant |
| `C_ess9_cos` | 0.9 | cosine |
| `D_ess5_cos` | 0.5 | cosine |

Each arm is scored under **three weight sets**: `live`, `ema999`, `ema9999`.

Held fixed across every arm: raw geometry, R11 scalar teacher, `smooth_laplace`
base kernel, 64 positives, 256 cloud, width 64, latent 32, Adam at 2e-3 peak,
CIFAR-32, disjoint train/eval splits.

**4 seeds × 15 000 steps.** Both are deliberate corrections to Phase 18B,
whose 2 seeds at 5 000 steps produced spreads over 100 FID and a control that
reversed sign. Raw arms cost ~176 s per 5 000 steps — 5–7× cheaper than
encoder arms — so the seeds and steps are affordable here. Estimated ~2.5 h.

Every arm shares a seed's generator initialization, latent stream and data
order, so all comparisons are **paired within seed**.

---

## 3. Declared formulas

Nothing here is swept; each is the standard form, fixed before the run.

**Cosine schedule** — `lr_t = 2e-3 · ½(1 + cos(π t / T))`, decaying to zero
at the final step.

**EMA with warmup** — `d_t = min(d, (1+t)/(10+t))`, then
`ema ← d_t·ema + (1−d_t)·θ`. The warmup stops the average from being pinned
to the random initialization early. Declared decays: **`ema999` (d = 0.999)
is the pre-registered primary**, an effective window of ~1 000 steps (≈7% of
training). `ema9999` (d = 0.9999) is reported for information only and **may
not be used to select the long-run configuration** — it is included because
it is free, not because it is a candidate.

---

## 4. Metrics

**FID is primary** (512 samples, Inception-V3 pool features), per the metric
audit. ED², spectral tail and second-moment ratio are recorded as
diagnostics and carry no decision weight. The moment-matched Gaussian bar is
recomputed inside the run from one seed stream.

---

## 5. What counts as an effect

Phase 16 measured a seed sd of 12.8 FID at 30 000 steps; Phase 18B showed
that spread grows sharply at shorter budgets. With 4 paired seeds this screen
is not entitled to call a small difference real.

**An effect is reported as real only if the paired difference has the same
sign in all 4 seeds.** Paired means are reported with their per-seed spread
in every case, including when the sign test fails. Any difference failing the
sign test is reported as unresolved, not as absence of an effect.

---

## 6. Declared outcomes

Written before the run; the reading follows whichever branch fires.

- **Both fixes help and compose** → the 232 figure was measuring a
  misconfigured system; the long run goes to the winning arm, and the
  program's absolute numbers to date are a floor rather than a ceiling.
- **Bandwidth helps, annealing does not** → the audit's ESS finding survives
  to a longer budget; the constant-step diagnosis of §F2 is a correct
  description that carries no cost, and the long run goes to ESS 0.5.
- **Annealing helps, bandwidth does not** → the ESS-0.5 advantage was
  specific to the audit's short budget and does not survive; §F1 is retired
  and the long run takes the annealing fix only.
- **Neither helps** → the recipe is at its ceiling in this harness, no cheap
  improvement remains, and the honest move is to run the long budget as-is
  and write up the mechanism results for what they are.
- **A fix helps at ESS 0.9 but reverses at 0.5 (or vice versa)** → the two
  interact; report the interaction and take the best *joint* cell, since the
  factorial measures it directly.

---

## 7. Scope, declared in advance

- One dataset (CIFAR-10, 32×32), one architecture, one metric with a floor
  near 71 and a bar near 247. Only arms measured identically are comparable;
  none is comparable with a published FID.
- Fresh seed block (`SEED_OFFSET = 32000`), disjoint from Phase 16 (28000),
  the invariance probe (30000) and Phase 18B (31000).
- This screen tests **training hygiene and one kernel knob**, not the
  encoder question. It changes no geometry and licenses no claim about
  encoder dependence.
- A multi-bandwidth (mixture) kernel — standard in the MMD generative
  literature and a genuine mechanism change — is deliberately **excluded**
  here so it cannot confound these two cheap fixes. It is a separate pass.
- Still not the paper's method: this harness drifts in pixel space using
  feature-space kernel weights.
