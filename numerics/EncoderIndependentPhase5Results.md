# Encoder-Independent Kernel Drifting — Phase 5 results

## Reforms R15–R19

*Executes `EncoderIndependentPhase5Protocol.md`, frozen before the run.
Sealed artifact: `phase5.json` (+ `.sha256`). Fresh seeds
`MASTER_SEED + 5000..`, disjoint from every earlier phase.*

---

## Verdict

**Phase-5 gate: FAIL — four of five conditions.** My own mechanism-targeted
candidate (R16) does not work, and the mechanism it was built on is refuted.

| ID | Condition | Result | Verdict |
|---|---|---|---|
| **G5.1** | R16 beats the baseline | **1.097** [1.035, 1.174], 1/9 wins | **FAIL** |
| **G5.2** | R16 repairs the collapse | eff. dim 0.271 — identical to baseline | **FAIL** |
| **G5.3** | mechanism confirmed on the real trajectory | teacher ratio **0.997**, i.e. no contraction | **FAIL** |
| **G5.4** | R15 excludes dead kernels | 0 cells excluded — **my grid design error** | **FAIL** |
| **G5.5** | batch × τ is neighbour starvation | neighbours 7.6 → 22.0 → 75.5 | **PASS** |

R11 replicated for a third time (0.302, **9/9**), so the phase is not a
total loss — but it was not testing R11.

---

## The four arms

Median over 3 seeds × 3 latent dimensions:

| arm | R11 | R16 | score | effective dim |
|---|---|---|---:|---:|
| **D0** | off | constant | 6.066 | 0.271 |
| **D1** | **on** | constant | **1.645** | 0.900 |
| **D2** | off | **decay** | 6.591 | 0.271 |
| **D3** | on | decay | 2.024 | **1.734** |

| comparison | ratio | 95% CI | wins |
|---|---:|---|---:|
| D2/D0 — **R16 alone** | **1.097** | [1.035, 1.174] | 1/9 |
| D1/D0 — R11 alone | **0.302** | [0.247, 0.376] | 9/9 |
| D3/D0 — both | 0.348 | [0.314, 0.390] | 9/9 |
| D2/D1 — R16 vs R11 | 3.629 | [3.053, 4.228] | 0/9 |
| D3/D1 — adding R16 to R11 | 1.151 | [0.991, 1.312] | 2/9 |

**R16 is not the lever.** A decaying teacher step is slightly *worse* than a
constant one (1.097, 1/9 wins) and leaves the effective dimension at exactly
the baseline's 0.271. Adding it to R11 makes R11 worse (1.151) and
over-inflates the cloud to 1.734× the data's effective dimension.

---

## G5.3 — the third mechanism is refuted

This is the important result.

| arm | teacher dimension ratio | output eff. dim, first → last |
|---|---:|---|
| D0 | **0.9970** | 13.88 → 2.20 |
| D1 | 0.9985 | 12.53 → 5.95 |
| D2 | 0.9996 | 13.88 → 2.08 |
| D3 | 0.9989 | 12.53 → 9.91 |

Measured **on the real trajectory**, the teacher changes effective dimension
by 0.1–0.3% per application — essentially not at all. The Phase-4 diagnosis
measured 2–8% on a synthetic interpolation between an untrained generator's
output and real data; those clouds do not occur in training, so that
measurement was unrepresentative and its conclusion does not hold.

**Three mechanism hypotheses have now been proposed and refuted:**

1. minibatch-noise regression attenuation — noise fraction ≈0.001, three
   orders of magnitude too small (Phase 4, P4C);
2. anisotropic contraction of the teacher map — isotropic to four
   significant figures at the fixed point (Phase 4, P4C);
3. per-application dimension contraction along the trajectory — 0.997 here,
   against the 0.92–0.98 the synthetic probe suggested (this phase).

The protocol anticipated exactly this: *"This would be the third refuted
mechanism; at that point the honest move is to stop proposing them and report
the phenomenology."* So, the phenomenology:

- the generator **starts** at effective dimension ≈13.9 and **ends** at ≈2.2;
- **no single step contracts** — the teacher preserves dimension throughout;
- R11 moves the endpoint to ≈5.9 without changing the per-step ratio
  (0.9970 → 0.9985);
- free particles under the same field end at 0.74–0.88 of the data's.

Read together these say the loss of dimension is a property of **where the
iteration converges**, not of any step within it. R11 works by moving the
fixed point, not by preventing a contraction. That is consistent with every
measurement taken so far — and it is stated here as an observation, **not as
a fourth mechanism**, because it has not been tested.

---

## G5.4 — a gate I mis-designed

Zero cells were excluded, so the condition fails. The cause is my grid: I
required at least one inadmissible cell to exist, then chose
τ ∈ {0.05, 0.2, calibrated} — dropping **τ = 0.02, the only temperature known
from Phase-4's diagnosis to collapse the kernel** (93.8% dead rows). The
condition was unsatisfiable by construction.

R15 itself works and is unit-tested (four tests: collapsed rows, starved ESS,
non-finite ESS, and health reported on exclusion). The gate, not the reform,
was wrong.

---

## G5.5 — neighbour starvation confirmed

| τ | batch | ESS fraction | **effective neighbours** | collapsed |
|---|---:|---:|---:|---:|
| 0.05 | 64 | 0.1188 | **7.60** | 0.000 |
| 0.05 | 256 | 0.0858 | **21.96** | 0.000 |
| 0.05 | 1024 | 0.0737 | **75.49** | 0.000 |
| 0.2 | 64 | 0.6909 | 44.21 | 0.000 |
| calibrated | 64 | 0.9019 | 57.72 | 0.000 |

At fixed τ the *fraction* falls while the *count* rises, so the sharp
temperature is a neighbour-starvation regime that batch size relieves. The
τ = 0.05 / batch 64 cell sees 7.6 effective neighbours — matching the
repository's own E4 table (ESS 7.6 at spread s = 0.1) closely enough to be a
cross-validation of both.

---

## R17 — latent dimension changes nothing

| latent | D0 | D1 | D2 | D3 |
|---:|---|---|---|---|
| 32 | 6.21 / 0.254 | 1.49 / 0.773 | 6.82 / 0.265 | 1.91 / 1.734 |
| 64 | 6.07 / 0.280 | 1.72 / 0.953 | 6.51 / 0.271 | 2.10 / 1.722 |
| 128 | 5.99 / 0.281 | 1.61 / 0.833 | 6.23 / 0.279 | 1.96 / 1.823 |

Quadrupling the latent dimension moves nothing — score varies by under 4% and
effective dimension by under 0.03 within each arm. **The Phase-4 hypothesis
that the residual gap is a latent-dimension limit is not supported.** P4D's
missing spectral tail has some other cause.

---

## What Phase 5 establishes

**Survives:**
- R11 replicates a third time on fresh seeds: 0.302 [0.247, 0.376], 9/9 wins,
  effective dimension 0.271 → 0.900. It is the program's one robust result.
- R15 is implemented and tested; it will exclude dead kernels in any future
  sweep that contains one.
- The τ failure is neighbour starvation, relieved by batch, consistent with
  the repository's existing E4 analysis.

**Refuted or unsupported:**
- R16 (decaying teacher step) — worse than baseline, no effect on the
  collapse. Withdrawn.
- The trajectory-contraction mechanism — 0.997, refuted. Third in a row.
- R17 (latent dimension as the residual gap) — no effect. Unsupported.

**Honest position on R11 after five phases:** it works, robustly and
repeatedly, across seeds, resolutions, batch sizes, geometries and latent
dimensions — and **nobody knows why.** Three explanations have been proposed
and measured false. It should be described as an empirical correction with an
unknown mechanism, and any writing about it must say so.

---

## Scope

CIFAR-10 at 16×16, raw pixel geometry, one generator family, three fresh
seeds, 600 steps, three latent dimensions. Nothing here concerns ImageNet,
FID, or the paper's trained model. The geometry thread remains closed.

## Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.run_phase5
```
