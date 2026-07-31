# Encoder-Independent Kernel Drifting — Phase 8 results

## Capacity is not it either: the generator fits its teacher 2.3× better and the cloud does not move

*Protocol: `EncoderIndependentPhase8Protocol.md`, frozen before the run.
Code: `run_phase8.py`. Artifact: `phase8.json` (+ `.sha256`), stdout in
`phase8.stdout.txt`. 148.9 minutes, 3 fresh seeds (`MASTER_SEED + 14000..`),
CIFAR-16, 600 steps, target ESS 0.9, field cloud 256, Adam/2e-3. Every cell
R15-admissible.*

---

## 0. Summary

| | prediction | outcome |
|---|---|---|
| **8A gate** | capacity could supersede R11 | **not passed — R11 survives a third time** |
| **8A secondary** | R11's advantage shrinks with width | **refuted** (see §2 — the flag says otherwise and the flag is wrong) |
| **8B** | realized teacher fraction rises with width | **confirmed — and it changes nothing** |

The finding is the conjunction of the last two:

> **Widening the generator 36× makes it fit its teacher 2.3× better
> (realized fraction 0.212 → 0.493) and leaves the cloud exactly where it
> was (second moment 0.354 → 0.331).** Under-fitting the teacher is
> therefore not the cause of the deficit. The problem is in **what the
> teacher asks for**, not in the generator's ability to deliver it.

This refutes the least-squares-shrinkage account as the operative term in the
real recipe — the gap the generator-contraction pass explicitly flagged as
live in its §7.

---

## 1. 8A — the controlling variable is flat across a 36× parameter span

| width | params | plain ED² | **plain 2nd** | R11 ED² | R11 2nd | tail (plain / R11) |
|---|---:|---:|---:|---:|---:|---|
| 32 | 36 387 | 1.1687 | **0.354** | 0.1958 | 0.941 | 0.0064 / 0.0503 |
| 64 | 109 635 | 1.1657 | **0.401** | 0.1386 | 0.949 | 0.0047 / 0.0501 |
| 128 | 366 723 | 1.0821 | **0.349** | 0.1830 | 1.017 | 0.0048 / 0.0575 |
| 256 | 1 323 267 | 1.1978 | **0.331** | 0.2955 | 1.112 | 0.0077 / 0.0614 |

Across all 12 uncorrected runs the second moment spans **0.267 – 0.533**, and
**0 of 4 widths** reach the band. The best uncorrected cell (1.0821) misses
the supersession ceiling of 0.1732 by **6.2×**. The gate does not fire.

Note the direction: the plain second moment goes 0.354 → 0.401 → 0.349 →
0.331. If anything it *falls* with capacity. `plain_second_moment_rises_with_width`
is **False**.

---

## 2. The trend flag fired and it is wrong — a defect in my instrument

`_trend_8a` reported **"capacity and R11 behave as substitutes"**. That
conclusion does not survive looking at the numbers.

| width | R11 ÷ plain ED² | 95% interval |
|---|---:|---|
| 32 | 0.165 | [0.108, 0.332] |
| 64 | 0.181 | [0.107, 0.498] |
| 128 | 0.172 | [0.169, 0.175] |
| 256 | **0.388** | **[0.193, 1.226]** |

Three problems, each disqualifying:

1. **The test is two-point.** `rising = ratios[-1] > ratios[0]` compares only
   the extremes and ignores that 32/64/128 are flat and mutually
   indistinguishable.
2. **The width-256 interval is enormous** — [0.193, 1.226] — and contains
   every other width's value. One noisy cell is carrying the verdict.
3. **Decisively: the ratio rose because R11 got *worse*, not because plain
   got *better*.** From width 64 to 256, R11 went 0.1386 → 0.2955 while plain
   went 1.1657 → 1.1978, i.e. unchanged. The prediction was that the plain
   arm would climb toward R11. **It did not move at all.**

So the secondary prediction is **refuted**, and the flag reports the opposite
because it measured a ratio without checking which side moved. This is the
third instrument defect recorded in this program (after Phase 7C's
monotonicity test on a unimodal relationship, and the spurious P2 flag in the
open-questions pass). Recorded rather than quietly re-read.

---

## 3. 8B — better teacher-fitting, identical cloud

The fraction of the requested teacher displacement the generator actually
realizes in one step:

| width | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|
| realized fraction | 0.212 | 0.313 | 0.358 | **0.493** |
| resulting 2nd moment | 0.354 | 0.401 | 0.349 | 0.331 |

Capacity does exactly what it should — the wide model tracks its teacher
**2.3× better** — and the cloud it produces is indistinguishable. That is the
cleanest disproof yet that the deficit is an under-fitting problem, and it
arrives by a completely different route from Phase 5's inner-steps result
(which found that forcing convergence to the teacher overshoots and does not
improve quality). Two independent methods, same conclusion.

---

## 4. Where this leaves the mechanism

The generator's second-moment deficit is now known to be invariant to:

| axis | span tested | phase |
|---|---|---|
| optimizer and learning rate | 3 optimizers × 60× lr | 6A |
| teacher step η | 100× (inert under Adam) | R24 |
| kernel bandwidth | realized ESS 0.82 – 0.99 | 7A |
| field cloud size | 64 – 512 | 7A |
| latent dimension | 8 – 512 | N4 |
| **model capacity** | **36× parameters** | **8A** |
| teacher-fitting quality | 2.3× realized fraction | 8B |

and it is an equilibrium rather than a failure to move: a free dilation
parameter converges *downward* (0.829) and the field's mean radial demand at
the converged model is +0.0004 (generator-contraction pass, N2).

Only one intervention has ever moved it: **changing what the teacher asks
for.** R11 does that, and nothing else tested does.

### What survives from the generator-contraction pass

- **N2 stands.** It measured the real recipe: the generator sits at its
  objective's optimum. Unaffected by Phase 8.
- **N3 stands as a fact about fixed-target regression** — point-target least
  squares is faithful at p/v ≥ 1 and shrinks below it. What Phase 8 refutes
  is the *inference* from that setting to the recipe, which the pass itself
  flagged as unclosed.
- The correction to Phase 4's `fit_to_free` figure stands (it was an
  underparameterization artifact).
- The re-reading of refuted hypothesis 1 — that the approximation-error term
  was never measured — **must now be withdrawn as an explanation of the
  recipe**: Phase 8 measured that term by varying capacity and it is not the
  driver.

---

## 5. The one thing that could have overturned all of this — checked

Every statement above says "equilibrium", and that rested on N2's two
measurements at 600 steps and nothing else: **no long-run second-moment trace
for the generator had ever been logged.** If the uncorrected generator were
still climbing at 600 steps, the deficit would be a *rate*, R11 an accelerant
rather than a repair, and §4's invariance table a list of things that do not
change a rate.

Measured directly — 6000 steps, 10× the budget every phase has used, plain
against R11, 3 seeds (`phase8_longrun.json`):

| arm | at 600 steps | at 6000 steps | late-window growth |
|---|---:|---:|---:|
| uncorrected | 0.433 | **0.472** | **−0.0342** |
| + R11 | 0.953 | 1.075 | +0.0205 |

**The equilibrium reading stands.** Ten times the budget moves the median
second moment from 0.433 to 0.472 — 9% — and the late-window trend is
*negative* in two of three seeds. The generator is not slowly climbing toward
the band; it plateaus near 0.47, which is still far outside `[0.7, 1.3]`.

Two honest qualifications:

- **Seed variation is real.** Seed 0 did climb (0.340 → 0.506, +49%) while
  seeds 1 and 2 were flat (0.433 → 0.472, 0.480 → 0.469). The median is the
  right summary but the spread is worth stating.
- **My verdict function is again miscalibrated.** It tests `|growth| < 0.02`
  and prints "still moving" for anything larger — so a *negative* growth of
  −0.034 reads as "not settled" when it actually means "wobbling, and
  certainly not climbing". The question asked was "is it still climbing?" and
  the answer is no. Fourth instrument defect recorded in this program;
  reported rather than re-read.

Consequence for the record: **600 steps modestly understates the converged
value** (0.401 → 0.472 at width 64, +18%). Every cross-arm comparison in
Phases 3–8 was at matched budget, so no conclusion changes — but absolute
uncorrected second moments quoted at 600 steps should be read as a slight
underestimate.

---

## 6. Scope

CIFAR-10 at 16×16, raw pixel geometry, paper Algorithm-2 field, target ESS
0.9, field cloud 256, one generator family, 3 fresh seeds, 600 steps,
Adam/2e-3. Widths beyond 256 untested. The anchor stays disabled and the
geometry thread stays closed. Nothing here concerns ImageNet, FID, or the
paper's trained model.

## 7. Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.run_phase8 `
  --stage all --seeds 3 --steps 600 --widths 32,64,128,256
```
