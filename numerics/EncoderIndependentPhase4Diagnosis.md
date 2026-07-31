# Encoder-Independent Kernel Drifting — Phase 4 failure diagnosis

*In-depth investigation of the three Phase-4 failures. Code:
`numerics/encoder_independent_drifting/diagnose_phase5.py`. Artifact:
`phase5_diagnosis.json`. Development seeds (`MASTER_SEED + 4000..`). Nothing
here feeds a gate. **One Phase-4 conclusion is corrected outright.**"*

---

## 0. Executive summary

| # | Phase-4 failure | Explanation found |
|---|---|---|
| **H4** | R11 fails at the paper's temperatures | **Fully explained, and my claim was mislabeled.** Those τ values collapse the kernel in raw pixel space — 93.8% of rows dead at τ = 0.02, median affinity 4e-20. R11 cannot fix a field carrying no information |
| **H2** | mechanism unknown after two refutations | **Third hypothesis supported.** The teacher *is* effective-dimension-contractive along the trajectory (0.92–0.98 per application, worst mid-way). The refuted P4C test missed it by probing only the endpoint |
| **H1** | why the generator and not the field | **Decomposed.** Least-squares point-target regression costs ~2× effective dimension under this capacity; the drifting self-reference adds ~1.5× on top |
| H3 | *(frozen-teacher probe)* | **Failed probe**, diverges by construction; reported, not interpreted |

The headline correction: **Phase 4's statement that "R11 fails at the paper's
declared operating point" is wrong.** It fails where the kernel is collapsed,
and applying the paper's τ numbers to raw 768-dimensional pixels collapses it.
The paper's τ grid is calibrated for *normalized encoder features*; using those
numbers in pixel space is not the paper's operating point, it is the paper's
numbers in a space they were never calibrated for.

---

## 1. H4 — the temperature failure is a collapsed kernel

Kernel health at each temperature, CIFAR-16 (768 ambient dimensions), batch
64. `ESS` is the mean row effective sample size as a fraction of the batch;
`collapsed` is the fraction of probes whose affinities have all underflowed.

| τ | bandwidth | ESS (init) | ESS (real) | median affinity | **collapsed rows** |
|---|---:|---:|---:|---:|---:|
| ESS-calibrated | 6.887 | 0.867 | 0.804 | 9.4e-02 | 0.000 |
| **0.02** | 0.356 | **0.020** | 0.027 | **4.0e-20** | **0.938** |
| **0.05** | 0.887 | **0.102** | 0.092 | 1.2e-08 | 0.000 |
| 0.2 | 3.555 | 0.667 | 0.536 | 1.1e-02 | 0.000 |
| 1.0 | 17.553 | 0.973 | 0.956 | 3.9e-01 | 0.000 |

At τ = 0.02 the kernel is **almost entirely dead**: 93.8% of probes have no
usable affinity at all, so the denominator *floor* — not the data — decides
their update. At τ = 0.05 the median affinity is 1e-08, i.e. an effective
nearest-neighbour rule with ESS ≈ 6.5 of 64.

This explains P4A's temperature cells without any new mechanism. R11 rescales
the teacher's second moment; when the teacher is built from a field that
carries no information, rescaling it changes nothing useful, and at τ = 0.05
it amplifies noise (ratio 1.12).

**The correction that follows.** The paper's grid (`numerics/README.md`,
Table 8 / A.6) is stated for encoder features normalized so the mean pairwise
distance is 1. I applied those numbers to raw pixels, where the same τ means
something entirely different. So P4A did not test the paper's operating point,
and Phase 4's conclusion must be narrowed to:

> R11 fails when the kernel is collapsed, and the paper's τ values applied to
> raw pixel space collapse it.

Whether R11 survives at the paper's *actual* operating point — those τ in a
normalized feature space — **remains untested.**

---

## 2. H2 — the teacher does contract, away from the fixed point

Both refuted P4C hypotheses share a defect worth naming: they probed the
teacher map **at the target law**, where the field is near zero by
construction and nothing can happen. Training spends almost all its time
elsewhere.

Repeating the measurement along an interpolation from an untrained
generator's output (α = 0) to a real sample (α = 1):

| α | eff. dim before | after | **ratio** | variance ratio |
|---:|---:|---:|---:|---:|
| 0.00 | 34.51 | 34.65 | 1.004 | 0.997 |
| 0.25 | 34.68 | 33.49 | 0.966 | 1.009 |
| 0.50 | 14.01 | 12.95 | **0.924** | 1.024 |
| 0.75 | 8.51 | 8.11 | 0.954 | 1.017 |
| 1.00 | 7.85 | 7.68 | 0.979 | 0.995 |

**The teacher map is effective-dimension-contractive by 2–8% per application,
worst at mid-trajectory** — and note the *variance* ratio is ≈1.0 throughout,
so this is a change in shape, not scale. That is exactly why the earlier
variance-based probes found nothing: they were measuring the wrong quantity in
the wrong place.

A 5% per-application loss compounds: `0.95^600 ≈ 0`. Free particles escape it
because they take a *decaying* fractional step (0.2 × (1 − t/T)), whereas the
regression replaces the output with the teacher at full strength every step.
That difference is the most promising untested lever (see R16).

This is the third mechanism hypothesis and the first with supporting
evidence. It is **not** confirmed: the measurement is on a synthetic
interpolation, not the real trajectory, and one seed.

---

## 3. H1 — decomposing the generator's contribution

Four regimes, one field, one architecture, median of 2 seeds:

| regime | score | ED² | **eff. dim ratio** |
|---|---:|---:|---:|
| free particles (no generator) | 1.61 | 0.142 | **0.835** |
| `fit_to_free` — least squares onto the converged free cloud | 3.45 | 0.588 | **0.444** |
| `self_teacher` — the ordinary recipe | 6.10 | 1.585 | **0.282** |
| *(`frozen_teacher` — failed probe)* | *1367* | *852* | *0.771* |

Two components, roughly multiplicative:

- **Least-squares point-target regression costs ~2×** (0.835 → 0.444). Fitting
  this generator to a *fixed, good* cloud already halves the effective
  dimension. Caveat: the probe asks 110k parameters to reproduce 512 × 768 =
  393k values, so it is underparameterized by 3.6× and this figure is partly a
  memorization artifact — but the drifting recipe *is* per-sample point-target
  regression, so the effect transfers in kind if not in size.
- **The drifting self-reference adds ~1.5×** on top (0.444 → 0.282).

R11 reaches ~0.87, *above* the `fit_to_free` figure, because it re-inflates
the teacher every step rather than fitting a fixed cloud — it is active
counter-pressure, not a better fit.

**`frozen_teacher` was a failed probe.** Computing the repulsion against a
frozen reference cloud makes it unbounded, and training diverges (ED² 852).
It is reported so the attempt is on record; it says nothing about the
mechanism.

---

## 4. Revisions to the record

| claim | status |
|---|---|
| Phase 4: "R11 fails at the paper's declared operating point" | **corrected.** It fails where the kernel is collapsed; those τ values collapse it *in pixel space*. The paper's real operating point is untested |
| Phase 4: "the mechanism is unknown" | **narrowed.** Three components now identified, one with direct support (H2) |
| Phase 4: "the self-mask is protective" | **stands**, and is now plausible: masking removes the strongest self-attraction term, which is the contractive part |
| Phase 3: R11 confirmed, 3.1×, 18/18 | **stands** |
| P4C's two refutations | **stand**, but are now explained: both probed the fixed point, where the map is provably benign |

---

## 5. Reforms for the next phase

### R15 — a kernel-health precondition on every sweep *(blocking)*

No cell may be reported as a method result if its kernel is collapsed. Any
temperature, bandwidth or resolution sweep must first check
`collapsed_row_fraction == 0` and `ess_fraction` within a declared band, and
**exclude** failing cells rather than scoring them. P4A scored three cells in
which the field was numerically dead and drew a conclusion about the paper
from them.

### R16 — damped regression, as an alternative to R11 *(highest value)*

The teacher map loses 2–8% of effective dimension per application (H2), and
the regression applies it at full strength every step while free particles —
which do not collapse — apply a decaying fraction. The obvious reform is to
regress toward `x + λ·η·V` with `λ < 1`, or to decay the step over training,
and compare against R11 head-to-head.

This is worth more than another R11 confirmation because it targets the
measured mechanism rather than compensating for its symptom, and because it
has no scale-matching side effect. If damping alone recovers the effective
dimension, R11 is superseded by something more principled.

### R17 — raise the latent dimension

P4D found the residual skyline gap looks like a missing spectral tail (real
data carries 13.4% of variance beyond the top 32 directions, the corrected
generator 4.7%), consistent with the 32-dimensional latent. H1 adds that the
generator is underparameterized for point-target regression. Sweep latent
dimension and width; it is cheap and both diagnostics point at it.

### R18 — measure the contraction on the real trajectory

H2 used a synthetic interpolation. Log the per-step effective-dimension ratio
of the teacher *during actual training*, for R11 on and off. That converts the
supported hypothesis into a confirmed mechanism, or refutes it as the previous
two were.

### R19 — test the paper's temperatures where they mean something

Re-run the temperature axis in a space where mean pairwise distance is
normalized to 1 — which is what the paper's grid assumes — so the question
"does R11 survive the paper's operating point" can actually be asked. Note
this requires a feature map, which puts it in tension with the
encoder-independence framing and should be stated as a scoping study, not a
method arm.

### Priority

**R15 and R18 first** (both are cheap and both concern the integrity of what
gets reported), then **R16** as the substantive candidate, then R17, then R19.

---

## 6. What this pass does not establish

> **H2 was subsequently REFUTED.** `EncoderIndependentPhase5Results.md`
> implemented R18 and measured the teacher's dimension ratio on the *real*
> training trajectory: **0.997**, not the 0.92–0.98 the synthetic
> interpolation suggested. Those interpolated clouds do not occur in
> training. R16, the reform built on H2, also failed (1.097, 1/9 wins). This
> is the third refuted mechanism; see the Phase-5 results for the
> phenomenology that replaces it.

- H2's contraction is measured on a synthetic interpolation at one seed; R18
  is what would confirm it.
- H1's ~2× capacity figure is inflated by an underparameterized memorization
  probe and should be read as a direction, not a magnitude.
- No reform proposed here has been implemented or tested.
- The paper's actual operating point remains untested, and nothing here
  concerns ImageNet, FID or its trained model.
