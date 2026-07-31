# Stage F1 results — the drifting map has one global attractor

## F1 FAILS its gate. But the finding is larger: real data does not survive the map either, and every start converges to the same rank-1.7 state.

*Protocol: `EncoderIndependentF1Protocol.md` v3. Pre-conditions: null
calibration GO (`f1_calibration.json`), all validation checks passed
(`f1_checks.json`), vetoes calibrated (`f1_vetoes.json`). Artifact: `f1.json`
(sha256 `f234a13c0d011b37…`), confirmatory. 3 units × 6 arms × 2 regimes ×
K = 20 000, 2.56 h.*

---

## 1. The gate

| unit | regime | recall @ K=200 | CI | passes |
|---|---|---:|---|---|
| 0 | replay | 0.0000 | 0.0000–0.0078 | False |
| 0 | stochastic | 0.0005 | 0.0000–0.0039 | False |
| 1 | replay | 0.0005 | 0.0000–0.0415 | False |
| 1 | stochastic | 0.0000 | 0.0000–0.0171 | False |
| 2 | replay | 0.0029 | 0.0000–0.0665 | False |
| 2 | stochastic | 0.0000 | 0.0000–0.0049 | False |

**F1 FAIL**, 0 of 3 units in both regimes. Every CI upper bound lies below the
0.05 gate. Per §9 this selects the **F3B prescribed-bridge branch**.

---

## 2. The larger finding: there is no good basin

Median effective rank and recall over 3 units × 2 regimes:

| start | K=0 | K=40 | K=200 | K=500 | K=1000 | K=20000 |
|---|---:|---:|---:|---:|---:|---:|
| **real_data** rank | 8.88 | 6.54 | 3.89 | 1.82 | 1.72 | **1.71** |
| real_data recall | 0.720 | 0.722 | 0.627 | 0.165 | 0.009 | 0.010 |
| **random_generator** rank | 45.87 | 33.81 | 5.70 | 1.89 | 1.69 | **1.68** |
| random_generator recall | 0.000 | 0.000 | 0.000 | 0.033 | 0.010 | 0.009 |
| **ambient_noise** rank | 438.19 | 435.27 | 6.81 | 1.92 | 1.66 | **1.68** |
| **ae_reconstruction** rank | 8.47 | 6.27 | 3.71 | 1.83 | 1.72 | **1.71** |
| ae_reconstruction recall | 0.264 | 0.253 | 0.176 | 0.029 | 0.011 | 0.004 |
| **trained_bad** rank | 6.17 | 6.08 | 3.34 | 1.81 | 1.73 | **1.73** |
| **basin_interpolation** rank | 6.81 | 5.84 | 3.01 | 1.86 | 1.76 | **1.75** |

> **Six starting distributions spanning effective rank 6.2 to 438.2 — real data,
> a random generator, white noise, autoencoder reconstructions, a trained
> collapsed cloud and an interpolation — all converge to rank ≈ 1.7 under both
> teacher regimes and three independent teacher seeds. 36/36 cells.**

The drifting map does not have a good basin and a bad basin. **It has one
attractor**, and it is a rank-1.7 collapsed state with recall ≈ 0.01 and KID
≈ 0.26 — worse than a moment-matched Gaussian.

---

## 3. This refutes Phase 26, and the refutation was a horizon artifact

`EncoderIndependentPhase26Results.md` concluded that "the data distribution IS a
stable attractor" and "the objective's fixed point is correct", from real data
holding recall 0.724 → 0.717 across 40 iterations. That measurement is
reproduced exactly here (C2: recall 0.71680 vs 0.717, KID +0.00041 vs +0.00041).

**The inference from it was wrong.** Effective rank declines monotonically from
step 0 — 8.88 → 6.54 → 3.89 → 1.82 — so the 40-step window sat at the top of a
decline that was already underway. Recall is the lagging indicator: it holds
near 0.72 through K = 40 and only breaks after K = 200.

So the program's diagnosis since Phase 26 — *"the target is right, the
amortization is wrong"* — is **false in both halves**. The target is not right:
real data decays out of it on a ~500-step timescale.

Two prior conclusions built on Phase 26 are consequently withdrawn:

- **Phase 27's basin map.** Its "basin edges" (blend 0.6, mixture 0.8) were
  measured at K = 40, inside the window where nothing has visibly collapsed yet.
  There is no basin boundary to measure.
- **§19/§21's F1 framing.** The critical path was built on reaching a good basin
  that does not exist.

---

## 4. What the run got right, and what it cost to find out

The finding only appeared because the horizon went to K = 20 000. At K = 200 the
control reads recall 0.627 and KID +0.006 — **entirely healthy**. The original
K = 200 protocol would have reported a valid control and an F1 fail, and the
conclusion would have been "the deployed start cannot reach the basin" rather
than "there is no basin". That is a materially wrong conclusion, and the only
reason it was avoided is that the horizon was raised on a cost measurement.

**A protocol defect this exposes, and it is mine.** §5 instructed that the
horizon be chosen *on cost alone*. Cost was the wrong sole criterion: the
positive control's validity range is a property of the map, it was never
measured, and choosing a horizon without it is how a 40-step artifact survived
five phases. **A validity-range measurement belongs in the pre-flight, beside
the null calibration.**

The gate itself was handled correctly. K = 200 was declared as the gate
checkpoint *from the control trajectory alone*, before any primary-arm outcome
past the K = 4 smoke was visible, which is precisely what a validity
precondition is for.

**Precision is again useless.** It holds 0.69–0.76 across the entire collapse,
including at rank 1.70. Fourth demonstration in this program.

---

## 5. Where this leaves the program

**F3B is selected**, but for a stronger reason than §9 anticipated. The declared
F1-fail reading was "the current drift cannot escape its bad basin *from this
initialization*". The measured reading is:

> The drifting map, as specified, has a unique global attractor with recall
> ≈ 0.01. No initialization reaches coverage because the map destroys coverage
> from every initialization, including from the data itself.

That removes the last hypothesis under which the current objective could have
worked. It also means the F3A branch — persistent trajectories `z → x_K(z)` —
is dead as designed: the endpoint map exists and is bit-reproducible, but every
endpoint lies in the collapsed attractor.

**What survives.** The one-step-equivalence, reproducibility and historical
regression checks all passed, so the harness is sound and the Phase-26 numbers
are real — it was the extrapolation from them that failed. The
encoder-independence result of Phases 17–18 is untouched and, if anything,
further insulated: no encoder choice matters to a map whose attractor is
collapse.

---

## 6. Scope

- 3 units × 6 arms × 2 regimes; units carry independent (source, teacher) seed
  pairs, so the convergence is not an artifact of one target realization.
- Gate at K = 200 with a valid control (recall 0.627 > 0.5); checkpoints beyond
  it are basin-structure characterization and were never gated.
- `u2 real_data stochastic` reported terminal recall 0.0776 with CI 0.0005–0.1148
  at rank 1.72 — recall-estimator noise at very low rank, not retained coverage,
  and it does not affect the gate.
- Recall CIs are conditional on the fixed evaluation reference (§6), and the
  null calibration bounds spurious exceedance at `p_null_upper` = 0.0149.
- Gate power at exactly 0.05 is ~44% under the 2-of-3 rule (§2.1's λ = 0.3
  control). A FAIL means "no coverage detected at this power". Given the
  measured effect — recall 0.009 at rank 1.68 — that caveat does not bind here.
- Still not the paper's method.
