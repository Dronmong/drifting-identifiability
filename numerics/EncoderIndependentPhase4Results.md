# Encoder-Independent Kernel Drifting — Phase 4 results

## Is the teacher contraction general, and is it mechanistic?

*Executes `EncoderIndependentPhase4Protocol.md`, frozen before the run.
Sealed artifacts: `phase4.json` (P4B, P4C) and `phase4_ad.json` (P4A, P4D),
each with a `.sha256`. Fresh seeds `MASTER_SEED + 3000..`, disjoint from every
earlier phase.*

---

> **Corrected in part by `EncoderIndependentPhase4Diagnosis.md`.** The claim
> below that R11 "fails at the paper's declared operating point" is **wrong**.
> Those τ values collapse the kernel in raw pixel space (93.8% of rows dead at
> τ = 0.02, median affinity 4e-20), and the paper's grid is calibrated for
> *normalized encoder features*, not pixels. The correct statement is that R11
> fails where the kernel is collapsed; the paper's real operating point remains
> untested. The diagnosis also supplies a supported mechanism — the teacher map
> loses 2–8% of effective dimension per application away from the fixed point,
> which both refuted P4C probes missed by testing only the fixed point.

## Verdict

**All three gated conditions FAIL.** The effect is real and reproduces
outside this package, but it is **narrower than Phase 3 suggested** and its
mechanism is **unknown** — two candidate explanations were proposed and both
were refuted by measurement.

| ID | Condition | Result | Verdict |
|---|---|---|---|
| **P4A** | R11 helps in every cell | 6 of 10 cells | **FAIL** |
| **P4B** | contraction reproduces cross-harness | reproduces, but only above ~4 dimensions | **FAIL** |
| **P4C** | regression attenuation predicts it | refuted; noise fraction ≈0.001 against a needed ≈0.75 | **FAIL** |
| P4D | residual-gap diagnostic | *(exploratory, no gate)* | — |

Phase 3 confirmed that R11 works. Phase 4 asked whether it describes
something general, and the answer is **partly** — with an important
exception at the paper's own operating point.

---

## P4A — R11 is robust, except at the paper's temperatures

| cell | R11 ratio | 95% CI | wins | eff. dim off → on | pass |
|---|---:|---|---:|---|---|
| base | 0.305 | [0.185, 0.421] | 3/3 | 0.303 → 0.819 | ✓ |
| batch 32 | 0.289 | [0.231, 0.400] | 3/3 | 0.235 → 0.978 | ✓ |
| batch 128 | 0.310 | [0.279, 0.347] | 3/3 | 0.298 → 0.995 | ✓ |
| resolution 32 | 0.300 | [0.222, 0.351] | 3/3 | 0.265 → 0.854 | ✓ |
| wavelet geometry | 0.299 | [0.280, 0.315] | 3/3 | 0.235 → 1.113 | ✓ |
| with anchor | 0.286 | [0.199, 0.391] | 3/3 | 0.319 → 0.772 | ✓ |
| **τ = 0.02** | **0.864** | [0.632, 1.109] | 2/3 | 0.164 → 0.793 | ✗ |
| **τ = 0.05** | **1.123** | [0.918, 1.392] | 1/3 | 0.182 → **0.160** | ✗ |
| **τ = 0.2** | 0.466 | [0.437, 0.497] | 3/3 | 0.284 → **0.273** | ✗ |
| **self-mask on** | 0.519 | [0.294, 0.750] | 3/3 | **0.521** → 1.129 | ✗ |

Two distinct failures, and they matter differently.

**At the paper's declared temperature grid, R11 stops working.** Across
τ ∈ {0.02, 0.05, 0.2} — the operating point in `numerics/README.md`, Table 8
/ A.6 — the ratio degrades to 0.86, 1.12 and 0.47, and at τ = 0.05 the
correction actively *hurts*. Decisively, at τ = 0.05 and τ = 0.2 the
effective dimension is **not restored** (0.182 → 0.160; 0.284 → 0.273), so
the correction is failing at its own stated job, not merely failing to help.
Every R11 result before this phase used an ESS-calibrated bandwidth, which is
*not* the paper's rule.

**The self-mask is itself partly protective.** With Algorithm 2's
`eye(N)*1e6` mask enabled, the uncorrected contraction is far milder
(effective dimension 0.521 against the unmasked baseline's 0.303) and the
uncorrected score is much better (2.5–4.8 against 5.2–8.2). This cell "fails"
only on the frozen precondition that the contraction be severe
(`< 0.50`) — but that is the informative part: **the paper's own recipe
mitigates the problem that R11 exists to fix.**

R11 remains robust across batch (32–128), resolution (16–32), geometry (raw
and wavelet) and the anchor, at ratios 0.286–0.310 with 3/3 wins throughout.

---

## P4B — it reproduces outside this package, with a clean dimension law

Using `numerics/lowdim_drift.py` **unmodified** — this repository's
independently audited verbatim Algorithm-2 port, cross-checked against
`driftlab.compute_v_paper`. Only the generator and training loop are new,
because that harness is particle-based.

Gaussian mixtures, batch 64, median over 3 seeds:

| dimension | eff. dim without R11 | with R11 | ED² without | ED² with | ED² ratio |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.998 | 0.997 | 0.0088 | 0.0238 | **2.71** |
| 4 | 0.917 | 0.954 | 0.0149 | 0.0123 | 0.83 |
| 8 | 0.659 | 0.846 | 0.0387 | 0.0057 | 0.15 |
| 16 | 0.391 | 0.637 | 0.0805 | 0.0088 | 0.11 |
| 32 | 0.212 | 0.373 | 0.1890 | 0.0145 | 0.08 |
| 64 | 0.096 | 0.168 | 0.3795 | 0.0280 | 0.07 |

The contraction **does** reproduce in a harness this program did not write —
which rules out an artifact of my field, kernel or generator. But it is
**strongly dimension-dependent**: absent at 2 dimensions (0.998), severe at
64 (0.096), with an approximately `1/d` decay above d ≈ 8.

There is a **crossover near d = 4**: below it R11 is unnecessary and mildly
harmful (2.7× worse ED² at d = 2); above it R11 is worth 7–14× on ED².

The gate required the contraction on ≥ 3 families with `< 0.50` uncorrected;
only the 16-dimensional mixture qualifies, so it fails as written. The gate
assumed dimension-independence — which the data show is false. That is a
mis-specified threshold, not a refutation, and the scaling law is a stronger
result than the binary would have been.

---

## P4C — the proposed mechanism is refuted, twice

The protocol committed in advance to an explanation: the teacher fluctuates
with the target minibatch, and least-squares regression converges to the
conditional mean, losing `Var(teacher | z)` per step. The prediction was that
the retention `1 − Var(teacher|z)/Var(teacher)` should track the measured
effective dimension.

| batch | measured noise fraction | predicted retention | measured eff. dim |
|---:|---:|---:|---:|
| 32 | 0.0009 | 0.9991 | 0.235 |
| 64 | 0.0005 | 0.9995 | 0.303 |
| 128 | 0.0004 | 0.9996 | 0.298 |
| 256 | 0.0002 | 0.9998 | 0.381 |

**Refuted.** The teacher's conditional noise is negligible — three orders of
magnitude too small. Predicted retention is ≈0.999 where ≈0.25 is needed.
Minibatch noise cannot be the cause.

A second hypothesis was then tested with evidence, as the protocol's failure
branch requires: perhaps the teacher map itself contracts, anisotropically,
hardest in directions the positive batch represents poorly. Measured on a
cloud already at the target law, per-direction variance ratio in the cloud's
own eigenbasis:

| dimension | top-quartile directions | bottom-quartile | anisotropy | eff. dim before → after |
|---:|---:|---:|---:|---|
| 2 | 0.9961 | 1.0019 | 0.99 | 2.00 → 2.00 |
| 16 | 0.9981 | 0.9993 | 1.00 | 7.43 → 7.44 |
| 64 | 0.9916 | 0.9939 | 1.00 | 31.88 → 31.87 |

**Also refuted.** One application of the teacher is essentially isotropic and
preserves effective dimension to four significant figures at every dimension.

**The mechanism is therefore unknown.** What is established: the collapse is
not minibatch-noise attenuation, not single-step anisotropic contraction, and
not a property of the field (free particles under the same field keep 0.74 of
the data's effective dimension). It emerges from the training trajectory of a
*parametric* generator, and beyond that this phase cannot say.

---

## P4D — the residual gap is a missing spectral tail

Covariance spectra, CIFAR-16 (exploratory):

| cloud | eff. dim | top-1 share | top-8 | top-32 |
|---|---:|---:|---:|---:|
| real data | 7.33 | 0.334 | 0.677 | 0.866 |
| corrected generator | **8.12** | 0.298 | 0.709 | **0.953** |
| skyline | 6.01 | 0.360 | 0.787 | 0.977 |
| free particles | 5.93 | 0.381 | 0.644 | 0.694 |

The corrected generator's effective dimension now slightly *exceeds* the
data's. What it lacks is the **tail**: real CIFAR carries 13.4% of its
variance beyond the top 32 directions, the corrected generator only 4.7%.
That is consistent with a 32-dimensional latent capping the representable
tail, and it suggests the residual skyline gap is a *latent-dimension* limit
rather than a second defect in the objective — testable by raising the latent
dimension, which this phase did not do.

---

## Corrections to earlier documents

| claim | status |
|---|---|
| Phase 3: "R11 improves the baseline 3.1×, 18/18 wins" | **stands** — reconfirmed at 0.286–0.310 across six independent cells |
| Phase-4 design: "the contraction is universal" | **narrowed.** It is dimension-dependent (absent below ~4 dimensions) and largely absent at the paper's declared temperatures |
| Phase-4 design/protocol: "regression attenuation explains it" | **withdrawn.** Refuted directly, as is the anisotropy follow-up |
| Phase-2 diagnosis: "the field is sound, the generator is the bottleneck" | **stands** — free particles keep 0.74 where the generator kept 0.25 |

---

## What Phase 4 establishes

**Survives:**
- The contraction is not an artifact of this package: it reproduces in the
  repository's independently audited Algorithm-2 harness.
- It follows a clean, tight dimension law (0.998 at d=2 to 0.096 at d=64,
  three seeds, very low spread), with a crossover near d=4 below which the
  correction is unnecessary and mildly harmful.
- R11 is robust across batch, resolution, geometry and the anchor.

**Does not survive:**
- The claim of universality. R11 fails at the paper's declared temperature
  grid, and at τ = 0.05 it makes things worse.
- The proposed mechanism, twice over.

**Newly learned:** the paper's own self-mask substantially mitigates the
contraction (0.521 against 0.303 uncorrected), so the regime R11 repairs is
one the paper's recipe already partly avoids.

## Scope

CIFAR-10 at 16–32 px and low-dimensional synthetic mixtures, one generator
family per harness, three fresh seeds, encoder-free throughout. Nothing here
concerns ImageNet, FID or the paper's trained model. The honest summary of
R11 after Phase 4 is: **a real and cross-harness-reproducible repair to
stop-gradient regression in higher dimensions at ESS-calibrated bandwidths,
of unknown mechanism, which does not transfer to the paper's declared
temperature grid.**

## Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.run_phase4
```
