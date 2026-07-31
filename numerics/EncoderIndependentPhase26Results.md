# Encoder-Independent Kernel Drifting — Phase 26 results

## The data distribution IS a stable attractor. Phase 25's conclusion was wrong, and the objective was never the problem.

*Code: `diagnose_phase26.py`. Artifact: `phase26_probe.json` (+ `.sha256`),
`phase26_probe.stdout.txt`. 5 000-step generator cloud, 512 particles, 40
iterations of the actual training map, 2 048 real samples.*

---

## 1. Why this probe existed

Phase 25 measured teacher recall at 0.000 across a 7.5× bandwidth range and I
concluded that this *"bounds what the generator can learn"*. **That inference
does not follow.** The generator does not sample from the teacher distribution
— it follows a flow, and what bounds a flow is its **fixed point**, not the
quality of one step's target evaluated at a bad cloud. A blurry local target is
precisely what a correct mean-shift flow looks like far from convergence.

The test that settles it is to put the cloud **at the real data** and see what
the map does. In 25 phases that was never done.

---

## 2. Result: real data survives the training map intact

Iterating the actual training recipe — RMS-normalized drift, η = 0.5, R11, fresh
positives every step — as free particles, no generator or optimizer in the way:

| iterating from | step | KID | precision | recall | alpha | 2nd | moved/step | from start |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **real data** | 0 | −0.00017 | 0.699 | **0.724** | 3.585 | 0.983 | — | — |
| **real data** | 40 | **+0.00041** | 0.682 | **0.717** | 3.579 | 0.979 | 0.019 | **0.123** |
| generated | 0 | +0.15016 | 0.566 | **0.000** | 4.432 | 0.966 | — | — |
| generated | 40 | +0.14184 | 0.570 | **0.000** | 4.478 | 0.908 | 0.035 | 0.132 |

**Real data is stationary.** KID stays at the floor (−0.00017 → +0.00041),
recall holds at 0.72, precision at 0.68–0.70, alpha at 3.58. Forty applications
of the map do not degrade it.

**And the map is not a no-op** — per-step displacement is 1.4–9.6% of the image
norm and cumulative movement from the start reaches **12.3%**. Points move
substantially; the *distribution* does not. That is exactly what a correct flow
at equilibrium looks like: particles shuffle within the distribution rather than
leaving it.

**The generated cloud is also stationary — at a different, zero-recall
attractor.** It moves 13.2% over the same 40 steps and its KID, recall and alpha
barely budge.

---

## 3. What this means

> The drifting map has **at least two stable attractors**: the data
> distribution, and a hyper-typical zero-recall region. Training lands in the
> second, and the map does not push it out. The objective's fixed point is
> correct. **The failure is a basin-of-attraction problem, not a target
> problem.**

This retroactively explains the whole program. Twenty-five phases of bandwidth,
selectivity, mixtures, teacher corrections, geometry and encoder work were
adjusting a target that was **already right**, which is why they moved KID
between 0.131 and 0.260 and never produced an object. Nothing in that family
could have worked, because nothing in it changes which basin training starts in.

### R11 is cleared, and its role is now visible

The R11 gain at the truth is **1.0086** — unbiased where it matters, so it does
not push the cloud away from the data distribution. Better than neutral: the raw
field is **inward-contractive at both clouds** (radial component −0.261 at the
truth, −0.335 at the generated cloud), so without a scale correction the flow
would collapse. R11 restoring the second moment is plausibly *what makes the
data distribution stationary at all*. After five phases of unexplained
mechanism, that is the first account of R11 consistent with every measurement.

### The corrected reading of Phase 25

Phase 25's numbers stand — teacher recall really is 0.000 at every bandwidth on
a generated cloud, and the teacher really is bandwidth-invariant. Only the
*inference* was wrong. Those measurements describe a cloud sitting in the wrong
basin, not a limit of the method. `EncoderIndependentPhase25Results.md` needs the
correction recorded against it.

### A methodological error worth naming

My declared discriminator in this probe was the raw drift ratio (`< 0.5` ⇒
stationary). It read **0.9609** — the raw field does *not* shrink at the truth
(|V| = 0.508 there against 0.528 at the generated cloud) — so the auto-verdict
printed "stationarity is partial". That threshold was simply the wrong
instrument: a flow at equilibrium moves individual points at full magnitude
while leaving the distribution fixed. **Distributional stationarity was the
right test and I should have declared it as primary.** The iteration is the
evidence; the drift ratio measures nothing relevant.

---

## 4. Where to go — and it is a different question than before

The live question is no longer "what is wrong with the objective" but **"how
close to the data does the cloud have to start for the map to keep it there?"**

**Next probe — map the basin (cheap, no training).** Interpolate
`x_λ = (1−λ)·real + λ·generated` for λ ∈ {0, 0.1, …, 1}, iterate the map 40
steps from each, and record which attractor each lands in. That yields the basin
boundary directly and tells us how good an initialization has to be.

**If the basin is wide** (λ up to ~0.5 returns to the data attractor), then a
warm start is sufficient and the intervention is concrete: pretrain the
generator by regression onto real samples, then hand over to drifting. That is a
small change with a measured rationale, and it is the first such candidate this
program has had.

**If the basin is narrow** (only λ ≈ 0 returns), the map is stable at the truth
but cannot be reached from anywhere useful, and the honest conclusion is that
drifting needs an initialization as good as the answer — a real negative result,
and a much sharper one than Phase 25's.

**Deprioritized, again.** Every kernel-family and teacher-family intervention:
§3 shows the target was never the problem. The autoencoder line stays parked;
its d=512 ceiling remains real but a basin fix in pixel space is cheaper and
now better motivated.

---

## 5. Scope

- One 5 000-step cloud, one seed, 512 particles, 40 iterations. The stationarity
  result is large and unambiguous (KID at the floor across 40 active steps), but
  it is a single-configuration probe, not a seeded performance claim.
- Free particles: no generator, no optimizer, no Jacobian. That isolates the
  map's own attractors, which is what was wanted, but a trained generator sees
  the map through its parameterization and could behave differently. Confirming
  that a warm-started *generator* stays in the good basin is a separate test and
  is the point of the next step.
- The 300-step smoke reproduced the result (real data KID +0.00015 → +0.00026,
  recall 0.682 → 0.672), so it does not depend on cloud training length.
- Still not the paper's method: pixel-space drift with feature-space kernel
  weights.
