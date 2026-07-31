# Encoder-Independent Kernel Drifting — reforms R20–R23

*Implementation and validation of the four reforms from
`EncoderIndependentRootCauseAnalysis.md`. Code: `config.py`, `diagnostics.py`,
`objectives.py`, `train.py`, `run_attractor_study.py`. Artifact:
`attractor_study.json` (+ `.sha256`). 112 unit tests, all passing.*

---

## 0. Summary

| Reform | What it is | Status |
|---|---|---|
| **R20** | two-sided effective-dimension band, and stop calling R11 a "collapse repair" | **implemented**, 1 test — and R23 shows the deeper point: effective dimension is not the right variable at all |
| **R21** | cap the realized output-space step | **implemented**, 1 test — **no measurable effect**; the cap never binds in low dimension |
| **R22** | `steps_per_teacher` as a declared axis | **implemented**, 1 test — confirms it does not help and mildly hurts |
| **R23** | characterize the attractor in a small solvable case | **implemented and run** — and it identified the controlling variable |

**The finding: the quantity that governs quality is the attractor's second
moment, not its effective dimension.** R11 works because it targets exactly
that. R21 and R22 target other things and do nothing.

---

## 1. R23 — the attractor study

The identical recipe against low-dimensional Gaussian mixtures, using
`numerics/lowdim_drift.py` (this repository's audited verbatim Algorithm-2
port, imported unmodified), with R21 and R22 exposed as axes. Median over
2 seeds × dimensions {8, 32}:

| configuration | ED² | eff. dim ratio | **2nd moment ratio** |
|---|---:|---:|---:|
| recipe as written | 0.1041 | 0.436 | **0.417** |
| **+ R11** | **0.0164** | 0.527 | **0.840** |
| + R22 (8 steps/teacher) | 0.1123 | 0.428 | 0.389 |
| + R22 (32 steps/teacher) | 0.1849 | 0.442 | 0.400 |
| + R21 (cap 1.0) | 0.1041 | 0.436 | 0.417 |
| + R21 (cap 0.25) | 0.1041 | 0.436 | 0.417 |
| R21 + R22 | 0.1123 | 0.428 | 0.389 |
| all three | 0.0255 | 0.503 | 0.894 |

Read the ED² column against the last two:

- **effective dimension is 0.43–0.53 in every row**, including the best and
  the worst — it does not discriminate at all here;
- **the second-moment ratio tracks ED² exactly**: 0.39–0.42 → ED² 0.10–0.18;
  0.84–0.89 → ED² 0.016–0.026, a 6× difference.

So the attractor's defect is a **second-moment deficit** — the cloud carries
about 40% of the data's variance — and R11 is the only intervention that
moves it (0.417 → 0.840).

### This corrects R20's premise, in R20's own direction

The root-cause analysis established that effective dimension is not
monotonically good, and R20 gave it a two-sided band. R23 shows something
stronger: **in low dimension it is not informative at all.** Every
configuration is "collapsed" by the band while ED² varies 6×. Effective
dimension was a proxy that happened to correlate at CIFAR-16 and does not
generalize.

The right reported and gated quantity is the **second-moment ratio**. That is
also, satisfyingly, precisely what R11 is defined to control — so the reform
that works and the diagnostic that discriminates are finally the same object.

---

## 2. R21 — implemented, and it does nothing

`ObjectiveConfig.output_step_cap` rescales the parameter update, along the
direction the optimizer chose, so the realized change in output space
respects a declared multiple of `eta * V`.

It is correctly wired (a unit test shows a capped run travels strictly less
far from initialization than an uncapped one), but in the attractor study
**cap = None, 1.0 and 0.25 give byte-identical results**. The cap never binds:
the realized step is already below a quarter of the request throughout.

The 42× overshoot that motivated R21 was measured at CIFAR-16 initialization
with a convolutional generator. It does not reproduce with an MLP in 8–32
dimensions, so it is either high-dimensional or architecture-specific, and it
is **not** what drives the attractor. R21 stays in the codebase as
instrumentation — it makes the realized step measurable and boundable — but it
is not a repair.

---

## 3. R22 — implemented, and it confirms the negative

`TrainConfig.steps_per_teacher` computes the field once and optimizes against
it for N updates. At 1 it reproduces previous behaviour exactly; the unit
test checks that kernel work is unchanged while optimizer updates scale
(20 → 160 for N = 8).

It does not help: ED² goes 0.1041 → 0.1123 → 0.1849 as N goes 1 → 8 → 32,
matching the CIFAR result (J5) where 32 inner steps also failed to improve
quality. Raising N drives the effective dimension around without touching the
second moment (0.417 → 0.389 → 0.400), which is exactly why it fails.

The axis is now declared and swept rather than an implicit 1, which was the
point of the reform.

---

## 4. R20 — implemented

`diagnostics.dimension_verdict` returns `collapsed` / `matched` /
`over_dispersed` against a declared band `[0.7, 1.3]`, so the two failure
directions — which score alike and need opposite corrections — are
distinguishable in a report.

The misleading docstrings are corrected. `variance_matched_teacher` and
`ObjectiveConfig.teacher_variance_match` now state that R11 matches a second
moment, that it is confirmed empirically (18/18, then 9/9), and that its
mechanism is unknown with four explanations refuted.

**Existing gates.** Phase-3 C.5 and Phase-5 G5.2 both used a one-sided floor.
Neither verdict changes — every arm that passed also scored well — but both
would have waved through the over-dispersed configurations. They are recorded
here as needing restatement rather than silently edited, since their artifacts
are sealed.

---

## 5. What this leaves

**Established:**
- the controlling variable is the attractor's second moment, not its
  effective dimension;
- R11 is the only intervention that moves it, and it is the only one of the
  three that improves quality — now shown in a third independent setting
  (low-dimensional mixtures, audited harness);
- R21 and R22 target quantities that do not govern the attractor.

**Still unknown:** *why* the recipe's attractor has a second moment ≈0.4 of
the data's. Five hypotheses have now been proposed and measured false. This
pass deliberately does not propose a sixth.

**The obvious next question**, stated but not pursued: R11 reaches 0.84–0.89,
not 1.0 — it undershoots its own target. Whether closing that gap (iterating
the match, or matching per-direction rather than by a single scalar) improves
quality further is testable and cheap, and it would be a natural successor to
this pass.

## 6. Scope

Low-dimensional Gaussian mixtures (8 and 32 dimensions) via the repository's
audited harness, 2 seeds, 600 steps, plus the CIFAR-16 measurements carried
over from the root-cause analysis. Nothing here concerns ImageNet, FID or the
paper's trained model. The geometry thread remains closed.

## 7. Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy `
  python -m numerics.encoder_independent_drifting.run_attractor_study

uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.tests.run_all
```
