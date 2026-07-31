# Encoder-Independent Kernel Drifting — root-cause analysis

*Deep investigation after three refuted mechanisms. Code:
`numerics/encoder_independent_drifting/diagnose_phase6.py`. Artifact:
`phase6_rootcause.json`. Development seeds (`MASTER_SEED + 6000..`, `+7000..`).
Nothing here feeds a gate.*

**This pass corrects a framing I have carried since Phase 2.**

---

## 0. Executive summary

Three mechanisms had been proposed and refuted, all asking what the *teacher*
does. The teacher turned out to be innocent (0.997 dimension ratio per
application), so the question was aimed at the wrong object. This pass asked
instead what the **optimizer** does with the field.

| # | Question | Answer |
|---|---|---|
| **J1** | When does the collapse happen? | **Suddenly, in the first ~70 steps.** 34.7 → 1.8, then flat for 530 more. Not a compounding process at all |
| **J3** | Is the field's requested change realized anisotropically? | **No.** Anisotropy is 0.43–1.61 with no consistent direction. Fourth hypothesis, also unsupported |
| **J4** | Does the generator actually reach its teacher? | **No.** With one optimizer step per teacher it closes 13% of the gap (residual 0.87). At 32 steps it closes 68% and the "collapse" disappears |
| **J5** | Does reaching the teacher improve *quality*? | **No.** Effective dimension goes to 1.4–3.3× the data's, coverage falls to 0.23–0.66, and the score does not improve |

**The correction:** I have been treating effective-dimension collapse as *the*
defect since Phase 2, and R11 as the thing that repairs it. J5 shows that is
wrong. Effective dimension is a **diagnostic with an optimum, not a quantity
to maximize** — and R11 works because it pins the second moment *to the
data's*, not because it raises dimension.

---

## 1. J1 — the collapse is an early transient, not a compounding process

Effective dimension of the generator output against training step:

| step | 0 | 70 | 140 | 210 | 280 | 350 | 420 | 490 | 560 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 34.7 | **1.8** | 2.1 | 2.4 | 2.1 | 1.9 | 2.1 | 2.0 | 2.1 |
| with R11 | 34.7 | 5.4 | 6.5 | 5.7 | 5.1 | 5.7 | 7.6 | 6.8 | 6.7 |

The entire collapse happens in the first ~70 of 600 steps, after which both
arms sit on a stable attractor. This retrospectively explains why all three
earlier mechanisms failed: **every one of them was a compounding-per-step
story**, and there is no compounding to find. The system reaches a fixed point
almost immediately and stays there.

R11 does not slow a decay; it selects a different attractor.

---

## 2. J3 — a fourth hypothesis, also unsupported

The stop-gradient loss has gradient `-2ηV/n`, so it is a device for injecting
`V`; the generator can only move along directions its Jacobian produces. The
natural hypothesis was that low-variance directions are under-realized.

Realized-over-requested change, resolved by eigendirection of the output
cloud:

| step | global | top quartile | bottom quartile | anisotropy |
|---:|---:|---:|---:|---:|
| 0 | **42.4** | 47.9 | 48.8 | 0.98 |
| 50 | 1.69 | 1.35 | 0.84 | 1.61 |
| 150 | 1.05 | 0.69 | 0.97 | 0.71 |
| 300 | 0.77 | 1.13 | 0.76 | 1.49 |
| 599 | 0.90 | 0.69 | 0.52 | 1.33 |

No consistent anisotropy (0.71–1.61, sign varies). **Unsupported** — the
fourth mechanism in a row.

But the *global* column is the interesting one. At step 0 the optimizer moves
the output **42× further than the field asked for**. The step size in output
space is not η; it is whatever Adam's per-parameter learning rate happens to
produce through the Jacobian, and early in training that is wildly off. This
is uncontrolled in the current recipe and coincides exactly with J1's collapse
window.

---

## 3. J4 — the generator never reaches its teacher

Matched total optimizer steps (600), varying how many are spent per teacher:

| inner steps | outer | residual to teacher | effective dimension |
|---:|---:|---:|---:|
| 1 *(the standard recipe)* | 600 | **0.869** | 2.13 |
| 8 | 75 | 0.693 | 2.52 |
| 32 | 18 | **0.323** | **19.46** |

With one optimizer step per teacher the generator closes only 13% of the gap
to its own target. Give it 32 and it closes 68% — and the effective-dimension
collapse vanishes entirely.

This is the first thing that *does* control the collapse, and it also
explains the J3 puzzle: the realized change has roughly the right *norm*
(0.77–1.05) while the residual stays at 0.87, so the update points in a
substantially different direction from the one requested. The generator moves
about as far as asked, but not where asked.

For a moment this looked like the answer.

---

## 4. J5 — but reaching the teacher does not help *(the correction)*

Median over 3 fresh seeds, matched total optimizer steps:

| inner | R11 | **score** | ED² | coverage | eff. dim ratio | residual |
|---:|---|---:|---:|---:|---:|---:|
| 1 | off | 7.479 | 2.02 | 0.953 | **0.271** | 0.955 |
| 8 | off | 6.382 | 1.49 | 0.953 | 0.298 | 0.725 |
| 32 | off | 6.757 | 1.46 | 0.926 | **1.396** | 0.305 |
| **1** | **on** | **1.773** | **0.169** | 0.971 | **0.902** | 1.003 |
| 8 | on | 1.947 | 0.162 | 0.906 | 1.348 | 0.526 |
| 32 | on | 2.999 | 0.296 | **0.664** | **3.348** | 0.223 |

Reading down the effective-dimension column against the score column:

- 0.27 → score 7.5 (bad)
- **0.90 → score 1.8 (best)**
- 1.40 → score 6.8 (bad)
- 3.35 → score 3.0 (bad)

**Effective dimension has an optimum at ≈1, not a monotone benefit.**
Restoring it by fitting the teacher harder (inner = 32) overshoots to 1.4–3.3×
the data's, collapses coverage from 0.97 to 0.23–0.66, and does not improve
the score. Adding inner steps to R11 makes R11 *worse* (1.77 → 3.00).

### What this corrects

Since Phase 2 I have described the defect as "variance collapse" and R11 as
its repair, and reported effective dimension as though more were better —
including in the Phase-3 gate condition C.5 (`ratio ≥ 0.60`) and the Phase-5
condition G5.2. Those gates are not *wrong* — every arm that passed them also
scored well — but the reasoning behind them was, and a one-sided floor would
have waved through the inner = 32 configurations, which reach ratio 1.4–3.3
and are bad.

The accurate statement is:

> R11 works because it pins the teacher's second moment **to the data's**.
> Matching is the mechanism; raising effective dimension is a side effect that
> happens to point the same way in the regime tested, and stops doing so as
> soon as the correction is applied harder.

---

## 5. Where this leaves the mechanism

Four hypotheses, all measured false:

1. minibatch-noise regression attenuation — noise fraction ≈0.001;
2. anisotropic contraction of the teacher map — isotropic to 4 s.f.;
3. per-application contraction along the trajectory — 0.997;
4. anisotropic tangent-space realization — no consistent direction.

What is now established rather than hypothesized:

- the collapse is a **fast transition to an attractor** (≈70 steps), not a
  decay;
- the generator **does not follow its teacher** (13% gap closure per step),
  and the update's direction differs from the request even when its magnitude
  does not;
- the output-space step size is **uncontrolled**, running to 42× the requested
  displacement at initialization;
- **which attractor** the iteration lands on is what R11 changes, and the
  right attractor is the one whose second moment matches the data.

So the object to explain is a *fixed point of the coupled
teacher-plus-partial-fit iteration*, not any per-step contraction. That is a
different kind of question from the four asked so far, and it is the one a
successor should ask.

---

## 6. Reforms

### R20 — report effective dimension as a two-sided quantity *(blocking)*

Every gate that uses it must have an upper bound as well as a lower one. The
declared band should be a ratio in `[0.7, 1.3]` of the data's, and any arm
outside it reported as mismatched in the direction it errs. Phase-3 C.5 and
Phase-5 G5.2 should be re-stated; neither changes its verdict, but both would
currently pass a configuration this pass shows to be bad.

Correspondingly, **stop describing R11 as "repairing variance collapse."** It
matches a second moment. That is what it does and it is what should be
written.

### R21 — control the step in output space *(highest value)*

The realized output change is currently whatever Adam produces — 42× the
request at initialization, and mis-directed by enough to leave 87% of the gap
open later. Rescale the parameter update so the realized change matches
`η·V` in norm, or clip it to a declared multiple. This makes η mean what the
plan says it means, and it targets the one quantity now shown to be both
uncontrolled and coincident with the collapse window.

It is also cheap to test: measure realized-versus-requested (J3's global
column) and rescale.

### R22 — inner steps as a declared axis

The recipe silently assumes one optimizer step per teacher. J4/J5 show that
choice sets the attractor, and that neither extreme is right. It should be a
declared, swept parameter, reported like batch and budget — not an implicit 1.

### R23 — study the attractor directly

The remaining question is what fixed point the coupled iteration selects. The
tractable version: take a small target (a 2-D or 8-D Gaussian mixture in
`lowdim_drift`), run the exact same recipe, and characterize the fixed point
analytically or by direct search. Four mechanisms have been refuted by
measuring the wrong object at CIFAR scale; a small solvable case is more
likely to yield the right one.

### Priority

**R20 first** (it is a correctness-of-reporting issue and affects two existing
gates), then **R21** as the substantive candidate, then R22, then R23.

Explicitly **not** recommended: another R11 variant, or a fifth per-step
mechanism hypothesis.

---

## 7. What this pass does not establish

- J5's comparison is 3 seeds at one resolution and one budget; the ordering is
  consistent across all three but the magnitudes are not confirmed.
- No reform here has been implemented or tested.
- The attractor account in §5 is a description of the measurements, not a
  mechanism. It has not been tested, and this pass deliberately stops short of
  proposing a fifth.
- R11 remains confirmed as an empirical correction (Phase 3: 18/18; Phase 5:
  9/9). What changes here is the *explanation* attached to it, not its
  standing.
- Nothing here concerns ImageNet, FID, or the paper's trained model.
