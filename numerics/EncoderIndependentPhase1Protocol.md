# Encoder-Independent Kernel Drifting — Phase 1 protocol

*Pre-outcome design for `EncoderIndependentKernelDriftingResearchPlan.md`
section 9, Phase 1. Every threshold below was frozen in
`numerics/encoder_independent_drifting/run_phase1_screen.py` and
`metrics.py` **before** the screen was launched; this document transcribes
them. Results go to `EncoderIndependentPhase1Results.md`.*

Entry condition: the Phase-0 exit gate passed
(`EncoderIndependentPhase0Results.md`, `phase0_gate.json`).

## 1. Question

> Can fixed compositional image geometry replace learned semantic geometry
> in drifting without sacrificing source-space correctness?

Phase 1 answers this at mechanism scale only. A negative answer stops or
redesigns the program before any CIFAR-10 or ImageNet compute is spent.

## 2. Frozen experimental design

| Item | Value |
|---|---|
| Targets | the 9 structured families in `datasets.suite()` |
| Seeds | 3 (`MASTER_SEED + 0,1,2`) |
| Steps | 300 |
| Batch (field role) | 64 |
| Controller / audit roles | 32 / 32, disjoint from the field role every step |
| Optimizer | Adam, lr 2e-3, 1 inner step |
| Generator | one `OneStepGenerator`, identical init across arms in a cell |
| Evaluation | 512 fresh latents vs 512 held-out target samples |
| Block combination | `sum` (fixed by the Phase-0 G0.3 measurement) |
| Bandwidths | target-only, calibrated to median row ESS = 0.5 |

The training budget was set from the **baseline** arm, not the candidate: a
convergence sweep showed A1 (raw pixel kernel-gradient) reaches full
coverage on the checkerboard target by 300 steps at lr 2e-3, and that higher
learning rates are strictly worse. A1 was still improving in precision at
900 steps, so this is a **matched-budget mechanism comparison, not an
asymptotic one**, and the results document must say so.

### 2.1 Paired randomness

Following the repository convention (`lowdim_drift.train`), within a
`(target, seed)` cell every arm receives the **same** generator
initialization, the same latent stream, the same target-minibatch stream,
the same cross-fitting role split and the same calibration sample. Arms
therefore differ only by their objective, and the gate's paired ratios are
not inflated by sampling noise that has nothing to do with the mechanism.

### 2.2 Known confound

The kernel-gradient field under a structured kernel is nearly orthogonal to
raw displacement (Phase-0 G0.4, cosine −0.002). A field that does not point
straight at the target needs a longer path to cover the same ground, so a
**fixed step budget structurally favours the direct raw-pixel field**. If A4
loses G1.1, that reading must be offered alongside "fixed geometry does not
supply the needed prior", and separated by a step-budget sweep before the
program is stopped.

## 3. Arms

| ID | Anchor | Geometry | Direction | Adaptation |
|---|---|---|---|---|
| A0 | none | raw pixel Laplace | standard | none |
| A1 | none | raw pixel smooth Laplace | kernel gradient | none |
| A2 | spectral | none | direct anchor gradient | none |
| A3 | none | wavelet | standard | fixed |
| A4 | none | wavelet | kernel gradient | fixed |
| A5 | spectral | wavelet | kernel gradient | fixed |
| A6 | spectral | random convolutional | kernel gradient | fixed |
| A7 | spectral | geometry dictionary | kernel gradient | adaptive |
| A8 | none | **locally trained encoder stand-in** | standard | fixed |

**A8 is not the plan's A8.** The plan specifies a pretrained paper encoder;
this repository has none and runs offline. A8 is a small convolutional
autoencoder trained from scratch on the same target family — if anything
*favoured* relative to a genuinely external encoder, since it sees target
data. It is context only and is excluded from every gate condition. See
`reference_encoder.py`.

## 4. Pre-registered normalized geometry score

Raw metrics are not comparable across target families, so the gate uses a
composite frozen in `metrics.py` before any arm ran:

```
score = geometric_mean over m in {ed2, sw1, patch_ed2, spectral_l1,
                                  off_support}
        of  m(generated, target_eval) / m(target_null, target_eval)
```

The denominator is the **target-vs-target null** at the same sample size:
what a fresh independent real sample scores. So `score = 1.0` means
"indistinguishable from real data under these metrics at this sample size",
and larger is worse. The four target pools (eval, null, calibration A,
calibration B) are independent draws shared by every arm in a cell.

Support precision/coverage uses a radius calibrated from the two
target-only calibration pools, so no arm can widen its own notion of "on
support".

## 5. Exit gate

The program continues to Phase 2 only if **all five** hold.

| ID | Condition | Threshold |
|---|---|---|
| **G1.1** | A4 materially beats A1 | geometric-mean paired score ratio ≤ 0.90 **and** bootstrap upper bound < 1 |
| **G1.2** | A5 stays close to A4 | paired ratio ≤ 1.10 |
| **G1.3** | A5 passes collisions A4 fails | A4's geometry blind on ≥ 1 case **and** A5's audit-bank anchor detects all 6 |
| **G1.4** | the anchor is practically present | median A5 anchor gradient share ≥ 0.05 for ≥ 25% of logged training, in ≥ half the cells |
| **G1.5** | gains hold broadly | A4/A1 ratio < 1 on a majority of targets **and** on every seed |

"Materially" (G1.1) is a ≥ 10% reduction in the geometric-mean paired ratio
with a paired-bootstrap upper bound below one — the repository's existing
paired-ratio convention.

G1.4 is the plan's anti-rhetoric gate (section 10.2): an anchor whose
gradient share stays below threshold for most of training and whose removal
changes no collision result is *rhetorically present but practically
absent*, and must be reported as such.

## 6. Declared failure branches

Per plan section 9, Phase 1:

- **If A4 fails to beat A1**, fixed wavelet geometry is not supplying the
  needed image prior. Test the convolutional kernel (A6) before considering
  a learned encoder. Do not tune wavelet hyperparameters to rescue the gate.
- **If A5 falls far short of A4**, the anchor is damaging optimization; the
  first repair is its weight, not its removal, and the trade-off must be
  reported.
- **If the anchor is numerically invisible** (G1.4 fails), the exact-zero
  argument is intact but the empirical claim of anchored correctness is not.

## 7. What Phase 1 may not conclude

Regardless of outcome, this screen cannot support:

- any statement about CIFAR-10, ImageNet, FID, or natural images;
- "encoder-free generation works" — 16×16 synthetic targets are a mechanism
  probe, not a generation benchmark;
- "the finite random-feature anchor is characteristic" — it is an unbiased
  estimator of an ideal expectation;
- "fixed features are cheaper" — the measured wall-clock says otherwise at
  this scale, and the cost ledger is reported with the result;
- any comparison to the paper's numbers, since neither the protocol nor the
  compute is matched.
