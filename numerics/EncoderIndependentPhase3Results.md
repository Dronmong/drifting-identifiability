# Encoder-Independent Kernel Drifting — Phase 3 results

## Corrected-baseline confirmation

*Executes `EncoderIndependentPhase3Protocol.md`, frozen before the run.
Sealed artifact: `phase3_confirmation.json` (+ `.sha256`), 65.1 min, 18 cells
(2 resolutions × 3 budgets × 3 seeds) × 3 arms + skyline + free-particle
references. Confirmation seeds `20261724-6`, disjoint from every development
seed.*

---

## Verdict

**Phase 3 exit gate: PASS — all seven conditions.**

Reform **R11** (teacher variance match) is confirmed on fresh seeds. It
improves the encoder-free baseline by **3.1×** (paired ratio 0.319,
**18/18** wins), holds at both resolutions, all three budgets and every
seed, and the mechanism it was derived from is confirmed: effective
dimension is restored from 0.264 to 0.867 of the data's, and the
generator-versus-free-particle quality gap closes from 3.44 to 1.09.

This is the **first confirmed positive result of the program.**

---

## Results

Median over 18 cells:

| arm | field | R11 | score | ED² | coverage | eff. dim ratio | parametric gap |
|---|---|---|---:|---:|---:|---:|---:|
| **C0** | paper Alg. 2 | off | 7.358 | 2.714 | 0.971 | **0.264** | **3.44** |
| **C1** | paper Alg. 2 | **on** | **2.514** | 0.334 | 0.914 | **0.867** | **1.09** |
| C2 | SNIS | on | 2.335 | 0.327 | 0.907 | 0.985 | 1.01 |
| *SKY* | *sliced Wasserstein* | — | *1.904* | *0.284* | *0.956* | *0.864* | — |

### Gate

| ID | Condition | Result | Verdict |
|---|---|---|---|
| **C.1** | material improvement | **0.319** [0.292, 0.351], **18/18** wins (threshold ≤ 0.50) | **PASS** |
| **C.2** | every seed | 0.314 / 0.322 / 0.322 | **PASS** |
| **C.3** | both resolutions | 16: 0.310, 32: 0.329 | **PASS** |
| **C.4** | every budget | 300: 0.327, 600: 0.315, 1200: 0.316 | **PASS** |
| **C.5** | variance collapse repaired | C1 **0.867** vs C0 0.264 (floor 0.60) | **PASS** |
| **C.6** | reaches the skyline | 1.140 [0.999, 1.307] (tolerance ≤ 1.25) | **PASS** |
| **C.7** | parametric gap narrows | C1 **1.09** vs C0 3.44 | **PASS** |

The effect is strikingly uniform: the paired ratio sits between 0.310 and
0.329 across every resolution, budget and seed. That is the signature of a
structural correction rather than a tuning artifact — R11 has no free
parameter to overfit.

---

## What the mechanism conditions establish

C.5 and C.7 were included so that a score improvement arriving *for a
different reason* would be caught rather than banked. Both confirm the
Phase-2 diagnosis exactly:

**Variance collapse is the defect, and R11 is its repair.** The uncorrected
baseline sits at 0.264 of the data's effective dimension — it occupies about
a quarter of the directions real CIFAR does. With R11 it reaches 0.867,
slightly *above* the sliced-Wasserstein skyline's 0.864.

**The parametric gap closes.** The Phase-2 diagnosis found that the same
field on free particles reached skyline quality while the generator lost
3.5×. Measured here on fresh seeds, C0's generator scores 3.44× worse than
free particles under its own field; C1's scores 1.09× — the generator now
realizes what the field asks for. That number is the diagnosis's central
claim, re-measured under confirmation conditions, and it lands where
predicted.

---

## Two secondary findings

**R12 is not needed once R11 is applied.** C2 (SNIS field + R11) versus C1
(paper Algorithm 2 + R11) is 1.014 [0.964, 1.064], 5/18 wins — a tie. The
paper's column reweighting was worth ~13% in Phase-2 conditions, where the
generator was collapsed; with the collapse removed it is worth nothing
measurable. R12 remains the correct implementation of the paper's estimator
and stays the default on those grounds, but it is not carrying the result.

**The skyline is reached, but not beaten.** C.6 passes at 1.140 with a
bootstrap interval whose lower end touches 1.0 (0.999) and 7/18 paired wins.
The honest reading is **parity, not superiority**: corrected encoder-free
drifting is statistically indistinguishable from a sliced-Wasserstein
objective on this testbed, and the Phase-2 claim that drifting is
fundamentally weaker is withdrawn. C1 also gives up some coverage relative to
both C0 and the skyline (0.914 against 0.971 and 0.956) while gaining an
order of magnitude on ED², which is worth watching.

---

## Corrections made during implementation

Two defects were found while building this phase and are recorded rather than
quietly fixed:

1. **The generator silently produced the wrong resolution.** `OneStepGenerator`
   upsamples by 2 from a 4×4 projection, so only sizes `4 · 2^k` are
   reachable, but it validated only divisibility by 4. Resolution 24 passed
   and emitted 32×32, surfacing as a matrix-shape error inside a kernel mid
   run. The generator now refuses unreachable sizes, with a regression test,
   and the protocol's second resolution was corrected from 24 to **32**
   before any Phase-3 result existed.
2. **R13's first implementation measured the wrong thing.** Comparing the
   *field residual* at the generator's output against the free cloud's is
   confounded: a collapsed generator sits in a dense region where the field is
   weak and therefore records a *lower* residual (0.965) than a healthy one
   (1.402). Field residual does not rank generators. R13 was reimplemented to
   compare the same *quality score* under both parametrizations, which is what
   the protocol intends and what C.7 now reports.

---

## Scope

- CIFAR-10 at 16×16 and 32×32, raw pixel geometry, one generator
  architecture, three fresh seeds, three budgets. Encoder-free throughout:
  no pretrained network is loaded and no class label is read by any
  objective, controller or metric.
- **Not** a claim about ImageNet, FID, or the paper's trained model, which
  uses a feature encoder at a different scale with a different batch and
  schedule and may not exhibit this contraction at all.
- The skyline is a distribution-matching reference at matched budget on the
  same generator, not a claim about generative quality in general.
- The geometry thread stays closed: Phase 3 asked nothing about wavelets,
  scattering or encoders, and nothing here reopens them.
- R11 is confirmed as a *repair to this implementation's transfer step*. It
  is a statement about stop-gradient regression onto a mean-shift teacher,
  not a proposed improvement to the paper's method.

## Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy `
  python -m numerics.encoder_independent_drifting.run_phase3_confirmation
```

CIFAR-10 must be present under `~/.cache/cifar`. 104 unit tests cover the
package.
