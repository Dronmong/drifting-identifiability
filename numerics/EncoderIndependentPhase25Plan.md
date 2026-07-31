# Encoder-Independent Kernel Drifting — Phase 25 plan

## Verification of the Phase 24 recommendation, and what replaces it

*Written before implementation, at the user's request. Verification probes are
recorded here; no runner has been built for any of it yet.*

---

## 1. My Phase-24 §6 recommendation is refuted. Do not implement it.

Phase 24 §6 proposed **per-row adaptive bandwidth** (t-SNE/UMAP perplexity
calibration), on the reasoning that the cloud cannot distinguish near from far
positives so the weights flatten. Two measurements kill it:

**The cloud sees near/far structure as well as real data does.**

| rows | within-row distance CV | nearest/median distance |
|---|---:|---:|
| real → positives | 0.2003 | 0.677 |
| cloud → positives | **0.2065** | 0.726 |

Ratio 1.03. The premise — "from a cloud point every positive is equidistant" —
is false.

**Row heterogeneity is negligible.** Realized ESS per cloud row: median 0.721,
q10 0.702, q90 0.741. A per-row calibration would equalize a spread of 0.04.

So per-row bandwidth would fix a problem that does not exist, and would
otherwise act as a smaller global τ — which is `C_sharper`/`E_sharper_pos`,
measured catastrophic at p = 0.0001. **Implementing it would have re-entered
the cliff by a different route.**

---

## 2. What the verification found instead: the calibration measures the wrong thing

```
tau from target-only calibration                     = 7.714
calibration ESS, diagonal included (what runs today) = 0.0500   <- hits target
calibration ESS, diagonal removed                    = 0.6019
cloud -> positives, realized                         = 0.7104
```

`calibrate_block_kernel` bisects τ so the **median row ESS of the
target-vs-target self-gram** equals the declared fraction. Every row of that
self-gram contains its own entry at distance zero, which dominates the row.
The target is therefore met by the self-match, not by selectivity among
distinct samples. Strip the diagonal and the same kernel on the same data reads
**0.6019**, essentially the 0.7104 the field actually operates at.

**The field never has a self-match** — it weights a cloud point against real
positives, all at positive distance. So the declared "target ESS" has never
described the field's behaviour, and the gap is a factor of ~12 (0.05 → 0.60).

The function does what its docstring says, so this is not a coding error. The
error is that the quantity it was built to declare is not the quantity that
governs the field, and no phase since Phase 7 — whose headline result was a
bandwidth reform — checked that they agreed.

**Scope of the correction.** Every `target_ess_fraction` in this program is
mislabelled. It does **not** invalidate Phase 22's conclusions, because Phase
22 reported *realized* ESS measured on the actual field (0.24–0.94) alongside
every arm, and its contrasts were computed on those runs. What changes is the
interpretation: the ladder labelled 0.5 / 0.05 / 0.01 was in fact exploring
realized 0.82 / 0.54 / 0.28, and the "sharpest viable" setting is far flatter
than any document has claimed.

---

## 3. What this does *not* fix, and the honest state of the mechanism

Correcting the calibration makes the knob honest. It does **not** buy
performance, because Phase 22 already measured the optimum in *realized* terms:

| arm | realized ESS | images averaged | KID |
|---|---:|---:|---:|
| F_mix | 0.707 | 45.2 | **+0.13116** |
| D_pos | 0.524 | 134.2 | +0.13634 |
| B_sharp | 0.536 | 34.3 | +0.14660 |
| A_control | 0.815 | 52.2 | +0.16084 |
| C_sharper | 0.279 | 17.8 | +0.25988 |
| E_sharper_pos | 0.237 | 60.6 | +0.26413 |

The optimum sits at realized 0.52–0.71 and both ends are worse. A corrected
calibration reaches the same operating point with an honest label.

**And the count of images averaged does not explain the ordering** —
`E_sharper_pos` averages 60.6 images and is the worst arm, while `B_sharp`
averages 34.3 and is fine. So "the teacher is a blur of N images" is not
sufficient either; something about *small τ itself* is harmful, independent of
how many images end up in the average. That is unexplained, and it is the gap
worth closing before any further mechanism change.

---

## 4. Concrete plan

### Step 1 — fix the calibration, as a correctness change (small)

Add a `exclude_self` option to the ESS measurement used by
`calibrate_block_kernel`, defaulting to the **new** behaviour, and record the
old behaviour in the docstring with the numbers above. Add a unit test pinning
the two values apart, so the distinction cannot silently regress.

Then re-derive the label for every Phase-22 arm: report `target_ess`,
`corrected_target_ess` and `realized_ess` side by side, so past documents stay
readable. **No new training run.** This is bookkeeping plus a test.

### Step 2 — the diagnostic that closes §3's gap (cheap, decisive, no long run)

Measure, as a function of τ, what the teacher actually *is*. For each of ~6
bandwidths spanning realized ESS 0.9 → 0.15, on a fixed cloud and fixed real
positives:

- **teacher sharpness** — spectrum alpha and precision/recall of the teacher
  images themselves (not of a trained generator);
- **teacher variance across batches** — recompute the teacher for the *same*
  cloud point against independent positive batches, and measure the spread.
  This is the candidate explanation for §3: small τ may raise target variance
  faster than it raises target sharpness, so the generator regresses onto a
  noisier target without a sharper one;
- **realized ESS**, under the corrected measurement.

Declared in advance:

- **Teacher variance rises sharply as τ falls while sharpness barely improves**
  → the cliff is a target-variance effect. The fix is variance reduction at
  fixed selectivity (more positives, teacher averaging across batches, or an
  EMA teacher), and that is a well-posed next experiment.
- **Teacher sharpness improves materially as τ falls** → the cliff is *not*
  about the teacher, so it lives in the optimization, and the next probe is the
  regression rather than the field.
- **Neither moves** → the mean-shift teacher form is the binding constraint at
  every bandwidth, and no kernel-family intervention can help. That would be
  the strongest possible negative and it would close the pixel-space line for
  good.

Cost: minutes. No training.

### Step 3 — only then, decide whether a long run is warranted

Nothing goes overnight until Step 2 names a branch. If Step 2's third branch
fires, the honest output is the write-up, not another run.

---

## 5. What I am explicitly not doing

- **Per-row adaptive bandwidth** — refuted in §1.
- **Latent-space drifting** — both rationales refuted in Phase 24 §3–4. The
  d=512 ceiling (KID 0.031, recognizable objects) stands and is worth
  returning to, but only behind a mechanism that gives the code space the
  convexity it lacks. A VAE is the candidate; it is not next.
- **Any further bandwidth or selectivity sweep** — §3 shows the optimum is
  already located in realized terms.
- **Any overnight run** — until Step 2 justifies one.

---

## 6. Standing caveats

- CIFAR-10 32×32; arms comparable only when measured identically.
- §1 and §2 are single-configuration probes on an untrained generator. The
  effects are structural and large (0.05 vs 0.60 on the same kernel), not the
  ~15 FID differences that need eight seeds — but a *performance* claim from
  any of this would need the full seed count.
- Still not the paper's method: pixel-space drift with feature-space kernel
  weights.
