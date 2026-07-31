# Encoder-Independent Kernel Drifting — open questions and next phase

*Methodical pass over the remaining unknowns, grounded in new measurement.
Code: `diagnose_secondmoment.py`. Artifact: `second_moment_study.json`.
Development seeds (`MASTER_SEED + 9000..`). Nothing here feeds a gate.*

**This pass found a single fact that retroactively explains five failed
reforms.**

---

## 0. The finding

`step_eta` — the parameter the plan's §6.5 objective is written around — is
**exactly inert under Adam.**

| optimizer | lr | η = 0.05 | η = 0.5 | η = 5.0 |
|---|---|---|---|---|
| **Adam** | 2e-3 | 0.6404 / 0.0301 | **0.6404 / 0.0301** | **0.6404 / 0.0301** |
| SGD | 5e-2 | 0.0898 / 0.2193 | 0.6467 / 0.0356 | 0.5810 / 0.1205 |

Independently, at CIFAR-16 through the real training loop, the relative
change in the generator's output from that same 100× change in η:

| optimizer | relative effect of 100× η |
|---|---:|
| Adam | 2.8 × 10⁻⁵ |
| SGD | 1.34 |
| SGD + momentum | 5.53 |

*(second-moment ratio / ED², 8-D Gaussian mixture, audited `lowdim_drift`
field, 400 steps.)*

A 100× range in η produces results **identical to every reported digit**
under Adam, and a large effect under SGD. The reason is immediate once
stated: the stop-gradient loss is `‖f − sg(f + ηV)‖²`, whose gradient is
`−2ηV/n`. **η enters only as a constant multiplier on the gradient, and Adam
is invariant to constant gradient rescaling.**

**Correction to an earlier phrasing.** A first draft of this document called
the Adam results "byte-identical". That overstated it: what was measured is
that the *reported statistics* agree to four decimals. Adam's update is
`lr·m/(√v + ε)`, which is invariant to a constant gradient rescale only up to
`ε`, so the invariance is near-exact rather than exact. The unit test added
with R24 measures the residual directly — a 100× change in η moves the
model's output by a relative **2.8 × 10⁻⁵** under Adam, against **1.34** under
SGD and **5.53** under SGD with momentum. That is a factor of ~48,000, which
carries the argument; "byte-identical" did not survive contact with the test
and is withdrawn.

### What this explains, retroactively

| earlier result | now explained |
|---|---|
| **R16** (decaying η schedule) did nothing — 1.097, 1/9 wins | η is inert; the reform was untestable as implemented |
| **RMS normalization** made no quality difference (Phase-1 D6) | another per-step rescale, nearly invariant under Adam |
| **R21** (output step cap) never bound | a third scale-based intervention |
| the geometry loss is pinned at η² (Phase-1 D1) | the loss value *and* its parameter are both uninformative |
| five refuted "mechanism" hypotheses | all five were about **magnitudes**; magnitudes are invisible to Adam here |
| **R11 works** | it is the only intervention tested that changes the gradient's *direction* — a per-sample rescale about the batch mean, not a global constant |

That last row is the unifying statement:

> Under Adam, only interventions that change the gradient **direction** can
> have an effect. Every scale-based intervention is invariant by
> construction. R11 works because it reshapes the target; η, RMS
> normalization and the step cap do not because they only rescale it.

**Scope.** This concerns the *neural-generator port* of drifting. The paper's
Algorithm 2 moves particles directly, with no optimizer, and there η is the
actual step. Nothing here says the paper's η is decorative — it says this
implementation's is.

---

## 1. The second-moment predictions were both refuted

The pass began by deriving the Gaussian fixed point. With RMS normalization
the field becomes `V ∝ x/σ`, whose magnitude does not vanish as `σ → s`, so
the scale should limit-cycle with an amplitude set by η. Two predictions
followed:

- **P1** — removing the normalization should let the second moment converge.
  **Refuted**: 0.409 (none) against 0.398 (rms), indistinguishable.
- **P2** — the deficit should grow with η. **Refuted**: identical to four
  decimals at η = 0.1, 0.5, 2.0. (The automated check reported `True` on
  floating-point noise below the fourth decimal; the numbers are flat and the
  flag is spurious — recorded so it is not mistaken for support.)

Both refutations have the same cause as everything else: η and the
normalization are invisible to Adam. **The derivation may well be right; the
experiment could not test it**, because the quantity it predicts about does
not reach the optimizer.

This is the sixth refuted mechanism, but it is the first that failed for a
*diagnosable* reason rather than an unknown one.

---

## 2. The open questions, ranked

### Q1 — what is the real step-size control, and has it ever been set?

Adam's learning rate is the only thing that sets how far the generator moves,
and it has never been swept properly: it was fixed at 2e-3 in the Phase-1
pre-registration on a three-point check, and every phase since inherited it.
Given that the parameter everyone thought was the step is inert, **the actual
step has been unexamined for six phases.** This is the cheapest and most
obviously overdue experiment in the program.

### Q2 — is direction-changing the general principle, and does a better direction change exist?

R11 is a *scalar* rescale about the batch mean — the crudest possible
direction change. If the principle is right, richer ones should do better:
per-coordinate or per-eigendirection moment matching, or whitening the
teacher toward the batch covariance. R11 also undershoots its own target
(0.81–0.89 rather than 1.0), which a per-direction version might close.

This is the natural successor to the only thing that works, and it is now
motivated by a stated principle rather than by trial.

### Q3 — does the paper's particle algorithm have this problem at all?

> **Answered in Phase 6C, and the answer is yes — this hypothesis is
> refuted.** Under a matched comparison the particle algorithm carries a
> second-moment deficit of 0.594 (constant step) / 0.627 (decaying), with no
> seed reaching the band. It is milder than the generator's 0.269 but real
> and consistent. The deficit belongs to drifting's dynamics, not to the
> neural port. See `EncoderIndependentPhase6Results.md` §3; the paragraph
> below is left as written, since it is what the reasoning looked like
> before the measurement.

The paper moves particles directly. There η *is* the step, Adam is absent,
and none of the invariance above applies. The free-particle runs in this
program consistently behave well (0.74–0.88 effective dimension, skyline-level
ED²). **The defect may be entirely an artifact of porting drifting onto a
neural generator with an adaptive optimizer** — which would be worth stating
plainly, and is testable by running the particle algorithm and the generator
port side by side under matched conditions.

### Q4 — should η stay in the objective?

Under Adam it cannot do anything. Keeping a parameter that provably has no
effect invites exactly the mistake R16 made. Either remove it, or document it
as inert and controlled by the optimizer's learning rate.

### Q5 — the anchor

Unchanged and still unresolved: ~3.5% on raw geometry, orthogonal to R11,
replicated 3/3, and the program's only source-correctness mechanism. It
should be folded into whatever final configuration is reported rather than
studied alone.

---

## 3. Proposed next phase

### Phase 6A — set the actual control variable *(do first, cheap)*

Sweep what really controls the step:

| axis | values |
|---|---|
| optimizer | Adam, SGD, SGD+momentum |
| learning rate | 5e-4, 2e-3, 8e-3, 3e-2 |
| R11 | off, on |

Report the second-moment ratio and ED². The question is whether the ~0.4
second-moment deficit is a *learning-rate* artifact that a proper sweep
removes — in which case R11 is compensating for a mis-set optimizer, which
would be the plainest possible explanation and has never been checked.

**Gate:** if any (optimizer, lr) without R11 reaches a second-moment ratio in
`[0.7, 1.3]` and ED² within 25% of the R11 arm, then R11 is superseded by
setting the step correctly, and the program's one positive result is
reinterpreted accordingly. That is a real possibility and the protocol should
say so before the run.

### Phase 6B — richer direction changes *(only if 6A does not dissolve R11)*

| arm | teacher correction |
|---|---|
| E0 | none |
| E1 | R11 — scalar second moment (the incumbent) |
| E2 | per-coordinate moment matching |
| E3 | full whitening toward the batch covariance |
| E4 | iterated scalar match (closes R11's undershoot) |

**Gate:** E2/E3/E4 beat E1 by ≥10% paired with a bootstrap upper bound below
1, on fresh seeds, with the second-moment ratio inside `[0.7, 1.3]`.

### Phase 6C — particle versus generator, matched *(the scoping question)*

Run the paper's particle algorithm and the generator port under matched
budget, batch and field, and report both. If the particle algorithm has no
second-moment deficit — as every incidental measurement so far suggests —
then the honest headline for this program becomes:

> Porting drifting onto a neural generator with an adaptive optimizer
> introduces a second-moment deficit that the particle algorithm does not
> have; a scalar moment match on the teacher removes it.

That is a narrower claim than "we improved drifting", and a more defensible
one.

### Implementation reforms to land alongside

- **R24** — document `step_eta` as inert under adaptive optimizers, or remove
  it. Add a test asserting the invariance so nobody re-derives it the hard
  way.
- **R25** — report the *optimizer* configuration in every artifact's gate
  summary. It is the real step control and is currently buried in
  `TrainConfig`.

### Priority

**6A first.** It is cheap, it tests whether the program's single positive
result is an artifact of an unswept hyperparameter, and no further mechanism
work is worth doing until that is settled.

---

## 4. What this pass does not establish

- The Adam-invariance claim is verified at 8 dimensions on one target family
  with two optimizers; the *mechanism* (Adam's per-parameter normalization) is
  standard and not in doubt, but the empirical check is narrow.
- The Gaussian fixed-point derivation in §1 is untested, not refuted — the
  experiment could not reach it.
- Q3's suggestion that the defect is a porting artifact is consistent with
  every incidental measurement but has never been the subject of a controlled
  comparison.
- No reform proposed here is implemented.
- Nothing concerns ImageNet, FID, or the paper's trained model.
