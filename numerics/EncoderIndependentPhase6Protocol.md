# Encoder-Independent Kernel Drifting — Phase 6 protocol

## Is R11 a repair, or a workaround for an unswept learning rate?

*Frozen pre-outcome design. Every threshold was fixed before any Phase-6 run.
Source: `EncoderIndependentOpenQuestions.md`. Results go to
`EncoderIndependentPhase6Results.md`.*

---

## 1. Why this phase exists

`step_eta` is **exactly inert under Adam** — a 100× range gives byte-identical
results, because it enters the loss only as a constant gradient multiplier and
Adam is invariant to that. The real step control is the optimizer's learning
rate, which was fixed at `2e-3` in the Phase-1 pre-registration on a
three-point check and inherited unexamined by every phase since.

So the program's one confirmed positive result — R11, replicated three times —
has never been compared against the possibility that the second-moment deficit
it corrects is simply a **mis-set optimizer**. That is the plainest available
explanation and this phase tests it first.

**This protocol can retire R11.** That outcome is declared in advance
(§5, 6A gate) and would be reported as the phase's result, not as a setback.

---

## 2. Phase 6A — sweep the real control variable

| axis | values |
|---|---|
| optimizer | Adam, SGD, SGD + momentum 0.9 |
| learning rate | 5e-4, 2e-3, 8e-3, 3e-2 |
| R11 | off, on |

Data: CIFAR-10 at 16×16, raw pixel geometry, paper Algorithm-2 field.
Seeds: 3 fresh, `MASTER_SEED + 10000..`. Budget 600 steps, batch 64.

Reported per cell: ED², the **second-moment ratio**, the effective-dimension
verdict (R20's two-sided band), coverage, and the optimizer configuration
(reform R25).

### 6A gate — the decisive one

> **If any (optimizer, learning-rate) cell *without* R11 reaches a
> second-moment ratio inside `[0.7, 1.3]` and an ED² within 25% of the best
> R11 cell, then R11 is superseded by setting the step correctly.**

If that happens, R11 is reinterpreted as having compensated for an unswept
hyperparameter, the Phase-3 and Phase-5 confirmations are re-scoped
accordingly, and Phase 6B is not run.

---

## 3. Phase 6B — richer direction changes *(only if R11 survives 6A)*

The unifying principle from the open-questions pass is that **only
interventions changing the gradient's direction can matter under an adaptive
optimizer.** R11 is the crudest such change — one scalar about the batch
mean. If the principle is right, richer ones should do better.

| arm | teacher correction |
|---|---|
| E0 | none |
| E1 | scalar second-moment match *(R11, the incumbent)* |
| E2 | per-coordinate second-moment match |
| E3 | whitening toward the batch covariance |
| E4 | scalar match with a declared gain of 1.2 |

E4 tests whether R11's undershoot (it reaches 0.81–0.89 rather than 1.0) is a
systematic gain deficit. The gain is **declared and reported, not selected**:
a single value, fixed here, with no search.

**6B gate:** an arm beats E1 by ≥10% on the paired v2 score with a bootstrap
upper bound below 1, on every seed, **and** lands inside the `[0.7, 1.3]`
second-moment band. An arm that improves the score while leaving the band is
reported as a trade, not a win.

---

## 4. Phase 6C — particle algorithm versus generator port

Every incidental measurement so far suggests the particle algorithm has no
second-moment deficit while the generator port does. This makes that a
controlled comparison: identical field, batch, budget and target, differing
only in whether a parametric generator is in the loop.

Reported, not gated — it is a scoping measurement whose purpose is to fix the
wording of any claim this program makes. If the particle algorithm is clean,
the defensible headline narrows to:

> Porting drifting onto a neural generator with an adaptive optimizer
> introduces a second-moment deficit that the particle algorithm does not
> have; a scalar moment match on the teacher removes it.

---

## 5. Implementation reforms landing with this phase

- **R24** — `step_eta` documented as inert under adaptive optimizers, with a
  unit test asserting the invariance so it is not re-derived the hard way.
- **R25** — optimizer and learning rate reported in every artifact's summary.
  They are the real step control and were previously buried in `TrainConfig`.

---

## 6. Declared failure branches

- **6A finds a clean no-R11 cell** → R11 is superseded; re-scope Phases 3 and
  5 and stop. Do not run 6B.
- **6A finds none** → R11 survives its sharpest test; proceed to 6B.
- **6B finds nothing better than E1** → the direction-change principle is
  right about *what works* but does not generalize to *better*; report R11 as
  the endpoint.
- **No tuning.** The learning-rate grid is declared above and is a sweep to be
  reported in full, not a search for the best cell to headline.

## 7. What Phase 6 cannot conclude

- Nothing about the paper's particle algorithm beyond the 6C comparison, which
  uses this repository's audited field, not the paper's full protocol
  (no CFG, no self-mask sweep, no encoder).
- Nothing about ImageNet, FID, or the paper's trained model.
- The geometry thread stays closed.
- A positive result is scoped to: *CIFAR-10 at 16×16, raw pixel geometry, one
  generator family, three fresh seeds, 600 steps.*
