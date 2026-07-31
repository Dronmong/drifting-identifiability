# Encoder-Independent Kernel Drifting — Phase 6 results

*Protocol: `EncoderIndependentPhase6Protocol.md`, frozen before the run.
Code: `run_phase6.py`, `objectives.py`, `train.py`, `config.py`,
`evaluate.py`. Artifact: `phase6.json` (+ `.sha256`), stdout in
`phase6.stdout.txt`. 116 unit tests, all passing. 24.2 minutes, 3 fresh
seeds (`MASTER_SEED + 10000..`), CIFAR-10 at 16×16, 600 steps.*

---

## 0. Summary

| stage | question | verdict |
|---|---|---|
| **6A** | is R11 just a workaround for an unswept learning rate? | **gate not passed — R11 survives**, decisively |
| **6B** | do richer direction changes beat the single scalar? | **gate not passed** — none beats it; one is decisively worse |
| **6C** | is the deficit an artifact of the neural-generator port? | **hypothesis refuted** — the particle algorithm has it too |

Phase 6 was designed to be able to retire the program's one positive result.
It did not: **R11 passed the sharpest test aimed at it so far.** The stage
that was meant to be a scoping formality (6C) is the one that overturned a
claim — mine.

---

## 1. 6A — the second-moment deficit is not a learning-rate artifact

Three optimizers × four learning rates × R11 on/off × 3 seeds = 72 runs.

| | uncorrected | + R11 |
|---|---|---|
| second-moment ratio, **every** stable run | **0.148 – 0.369** (32 runs) | **0.853 – 1.195** (18 runs) |
| cells landing in the declared band `[0.7, 1.3]` | **0 of 12** | 6 of 6 that converged |
| best cell, median ED² | 1.3696 (SGD, 5e-4) | **0.1205** (SGD+momentum, 5e-4) |

The gate asked whether any uncorrected cell could reach the band with an ED²
within 25% of the best R11 cell — a ceiling of 0.1507. The best uncorrected
cell misses it by **9×**, and not one uncorrected cell of the twelve came
near the band.

**A 60× range of learning rate across three optimizers moves the second
moment from 0.148 to 0.369. R11 moves it to ~1.0.** The plainest available
alternative explanation for the program's one positive result is dead, and
the deficit is confirmed as a property of the recipe rather than of an
unswept hyperparameter.

Learning rate is not irrelevant, though — it matters *once R11 is on*. Adam
at 3e-2 degrades to ED² 0.396 against 0.158–0.205 at the three lower rates,
and the best cell in the whole grid (SGD+momentum at 5e-4, 0.1205) is about
30% better than the incumbent configuration Adam/2e-3 inherited from Phase 1
(0.1688). Sweeping the real control variable was worth doing; it just did not
do what it was hypothesized to do.

### A new constraint: R11 needs an optimizer that normalizes magnitude

**18 of the 36 R11 runs diverged** — every SGD and SGD+momentum cell at
lr ≥ 2e-3, none under Adam at any rate. Four uncorrected runs also exploded
at lr = 3e-2 (second moments up to 1.06e7, reported rather than dropped).

That is the same fact as R24 seen from the other side. A corrected teacher
sits far from a collapsed generator's output, so it presents a large
gradient; Adam's per-parameter normalization absorbs it, plain SGD does not.
**The gradient-magnitude blindness that makes Adam ignore `step_eta` is what
makes it able to use R11 at all.**

---

## 2. 6B — resolving more directions does not help

The unifying principle from the open-questions pass was that only
interventions changing the gradient's *direction* can act under an adaptive
optimizer. The natural extrapolation — that a correction resolving more
directions should act more — is **wrong**.

| arm | correction | median ED² | 2nd moment | ratio vs E1 | 95% interval | wins |
|---|---|---:|---:|---:|---|---:|
| E0 | none | 1.7621 | 0.256 | 3.371 | [1.887, 4.647] | 0/3 |
| **E1** | **scalar (R11)** | **0.1688** | **1.080** | 1.000 | — | — |
| E2 | per-coordinate | 0.1888 | 0.961 | 1.095 | [0.652, 1.420] | 1/3 |
| E3 | eigendirection | 0.2403 | 0.775 | 1.130 | [0.588, 1.659] | 1/3 |
| E4 | scalar, gain 1.2 | 0.5359 | 1.445 | 1.975 | [1.050, 2.898] | 0/3 |

The declared ratio cap **never bound** (0.0 in every arm, every seed), so the
numerical guard is not a confound in any of this.

- **E2 and E3 show no evidence of improvement.** Their intervals straddle 1
  and are wide on 3 seeds, so the honest reading is "indistinguishable from
  the scalar", not "worse than it".
- **E4 is decisively worse** — its interval excludes 1. This settles the
  question the R20–R23 pass left open. R11's undershoot to 0.81–1.08 is
  **not** a systematic gain deficit waiting to be closed: over-correcting to
  1.445 costs a factor of 2 in score. Near 1.0 is where it should be, and
  the scalar match already gets there.

So the direction-change principle explains **what is able to act**, not
**what acts better**. One scalar about the batch mean is, so far, the whole
of the effect.

---

## 3. 6C — my scoping hypothesis was wrong

The open-questions pass proposed (Q3) that the deficit "may be entirely an
artifact of porting drifting onto a neural generator with an adaptive
optimizer", on the strength of free-particle runs that had always looked
healthy. Under a matched comparison — identical field, kernel, batch, budget
and target — that is **false**.

| regime | median ED² | second moment | in band? |
|---|---:|---:|---|
| particles, constant step | 0.3446 | **0.594** | no |
| particles, decaying step | 0.2751 | **0.627** | no |
| generator, uncorrected | 1.6672 | 0.269 | no |
| **generator + scalar match** | **0.1601** | 1.067 | **yes** |

The particle algorithm carries a second-moment deficit of its own — 0.58,
0.59, 0.60 across the three seeds, with no overlap with the band. It is
**milder** than the generator's, roughly 0.6 against 0.27, but it is real and
perfectly consistent.

The correct statement is therefore narrower than "we improved drifting" but
wider than the port-artifact story I proposed:

> The second-moment deficit belongs to the kernel drifting dynamics
> themselves — the particle algorithm carries about 60% of the data's
> variance. Porting onto a neural generator roughly halves it again, to 27%.
> A scalar moment match on the teacher removes it, and is worth ~10× on ED².

Note also that the corrected generator **beats free particles** under matched
budget (0.160 against 0.275–0.345), which the uncorrected generator does not
come close to doing (1.667).

**Caveat, stated rather than glossed:** no particle trace was logged, so
these runs cannot distinguish "settled at 0.6" from "still growing at step
600". The decaying-step arm ends slightly *higher* than the constant-step arm
(0.627 vs 0.594), which is weak evidence against simple incompleteness, but
it is weak. Logging the trace is the obvious next measurement and it is
cheap.

---

## 4. Reforms landed

| reform | what | status |
|---|---|---|
| **R24** | `step_eta` documented as inert under an adaptive optimizer | **implemented**, 1 test |
| **R25** | optimizer and learning rate configurable and reported in every artifact | **implemented**, 1 test |
| **R26** | teacher corrections generalized: scalar / per-coordinate / eigendirection, with a declared gain and a measured guard | **implemented**, 3 tests |

**R24's test corrects an overstatement of mine.** The open-questions pass
called the Adam results "byte-identical"; they are not. Adam's update is
`lr·m/(√v + ε)`, invariant to a constant gradient rescale only up to `ε`.
Measured through the real training loop, a 100× change in η moves the output
by a relative **2.8e-5** under Adam, **1.34** under SGD, **5.53** under
SGD+momentum. A factor of ~48,000 carries the argument; the word
"byte-identical" did not survive the test and is withdrawn. Both documents
are corrected.

`variance_matched_teacher` now delegates to `corrected_teacher`, so R11 is
one declared mode of a general facility rather than a special case.

---

## 5. What Phase 6 establishes

**Established:**
- the second-moment deficit is **not** an unswept-learning-rate artifact —
  0 of 12 uncorrected cells reach the band across a 60× rate range and three
  optimizers;
- R11 survives its sharpest test and remains the program's only confirmed
  positive, now in a fourth independent setting;
- the deficit is **present in the particle algorithm too** (~0.6), so it is a
  property of drifting's dynamics, not of the neural port;
- richer direction changes do not beat one scalar, and deliberate
  over-correction is decisively worse — R11's "undershoot" is not a deficit;
- R11 requires an optimizer that normalizes gradient magnitude: it diverges
  under plain SGD at lr ≥ 2e-3 in every cell tested.

**Still unknown:** *why* the drifting dynamics land at ~0.6 (particles) and
~0.27 (generator) of the data's variance. Six mechanism hypotheses have now
been refuted. Phase 6 did not propose a seventh; what it did is move the
question off the generator, where five of those six were aimed, and onto the
field dynamics, where the deficit demonstrably already exists.

**The most promising next measurement**, from 6C: log the particle trace and
find out whether 0.6 is an attractor or an unfinished trajectory. If it is an
attractor, the mechanism question can be asked in the particle setting —
no generator, no optimizer, no parametrization — which is far simpler than
anything the last six phases have had to work with.

## 6. Scope

CIFAR-10 at 16×16, raw pixel geometry, paper Algorithm-2 field,
ESS-calibrated bandwidth, one generator family, 3 fresh seeds, 600 steps.
Nothing here concerns ImageNet, FID, the paper's declared temperature grid,
or the paper's trained model. The geometry thread remains closed. The anchor
was not enabled in any Phase-6 arm.

## 7. Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.run_phase6 `
  --stage all --seeds 3 --steps 600 --resolution 16

uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.tests.run_all
```
