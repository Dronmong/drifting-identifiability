# Encoder-Independent Kernel Drifting — Phase 3 protocol

## Corrected-baseline confirmation

*Frozen pre-outcome design. Every threshold below was fixed before any Phase-3
run. Results go to `EncoderIndependentPhase3Results.md`.*

---

## 1. Why this phase exists

`EncoderIndependentPhase2Diagnosis.md` found that the drifting generator had
collapsed to effective dimension 2.34 against the data's 8.32, that this
single defect accounted for the entire gap to a sliced-Wasserstein skyline,
and that one scalar correction (**R11**) removes it — taking the composite
from 6.69/7.05 to 1.92/1.30 and reaching parity with the skyline.

That was **development evidence**: two to three seeds, one resolution, one
budget, and the correction was adopted *after* seeing it work. The
repository's standing discipline is that such a finding must pass a fresh
frozen test before it is called a result. This phase is that test.

The geometry thread stays closed. Phase 3 asks nothing about wavelets,
scattering or encoders. It asks whether the **corrected encoder-free
baseline** holds up.

---

## 2. Question

> With the teacher variance match (R11) and the paper's real Algorithm-2
> field (R12), does encoder-free raw-pixel drifting hold its improvement
> across unseen seeds, two resolutions and three budgets — and does it reach
> the sliced-Wasserstein skyline?

---

## 3. Reforms carried in

| ID | Reform | Status entering Phase 3 |
|---|---|---|
| **R11** | teacher variance match | implemented, opt-in, 3 tests |
| **R12** | `direction_mode="paper"`, the real Algorithm 2 | implemented, verified against `lowdim_drift.drift_paper`, 2 tests |
| **R13** | parametric zero-set reachability | **implemented for this phase** |
| **R14** | effective dimension reported every run | **implemented for this phase** |

### R13 — measure the zero-set *through the generator*

Phase-2's entry gate certified that each field's zero-set was reachable, but
it tested a **free particle cloud**, which can take any configuration. The
generator's reachable set is a low-dimensional manifold, and the entire
Phase-2 failure lived in that difference. R13 reports both residuals and
their ratio:

```
parametric_gap = residual(generator) / residual(free particles)
```

A gap near 1 means the generator can realize what the field asks for. The
Phase-2 configuration should show a large gap and the R11 configuration a
smaller one; if it does not, the mechanism story in the diagnosis is wrong.

### R14 — effective dimension as a standing diagnostic

Participation ratio of the generated cloud's covariance spectrum, reported
against the real data's. It is one line, it is cheap, and it made a 3.5×
defect obvious after four phases of not looking for it.

---

## 4. Frozen design

| Item | Value |
|---|---|
| Data | CIFAR-10, disjoint `train`/`eval` splits, no labels anywhere |
| Resolutions | 16 and 32 |
| Budgets | 300, 600, 1200 steps |
| Seeds | 3, **fresh** — offset 1000 from every seed used in development |
| Batch / controller / audit | 64 / 32 / 32, disjoint every step |
| Geometry | raw pixel only |
| Score | `normalized_geometry_score_v2`, null averaged over 5 draws |

**Seed discipline.** The R11 development runs used `MASTER_SEED + 0,1,2`.
Phase 3 uses `MASTER_SEED + 1000,1001,1002`. No development seed is reused,
and no threshold below was chosen after seeing a Phase-3 number.

**Pre-run correction (resolutions).** This protocol first specified 16 and
24. Implementation revealed that the fixed generator upsamples by 2 from a
4×4 projection, so it can only produce sizes `4 · 2^k`; 24 passed the old
divisibility check and silently emitted 32×32. The generator now refuses
unreachable sizes (with a regression test) and the second resolution is
**32**. The correction was made before any Phase-3 result existed and
changes no threshold.

### Arms

| ID | Field | R11 | Purpose |
|---|---|---|---|
| **C0** | paper Algorithm 2 | off | the Phase-2 baseline; the control |
| **C1** | paper Algorithm 2 | **on** | the candidate |
| **C2** | SNIS mean shift | on | does R12 still matter once R11 is applied? |
| **SKY** | sliced Wasserstein | — | skyline; never a candidate in a gate |

Matched within a cell: generator initialization, latent stream, target
minibatch stream, calibration sample, budget, batch sizes, optimizer. Arms
differ only by field and by R11.

---

## 5. Exit gate

| ID | Condition | Threshold |
|---|---|---|
| **C.1** | R11 materially improves the baseline | paired C1/C0 ratio ≤ 0.50, bootstrap upper bound < 1 |
| **C.2** | the improvement is not one seed | C1/C0 < 1 on **every** seed |
| **C.3** | it holds at both resolutions | C1/C0 < 1 at 16 **and** 24 |
| **C.4** | it holds at every budget | C1/C0 < 1 at 300, 600 **and** 1200 |
| **C.5** | variance collapse is actually repaired | C1 effective-dimension ratio ≥ 0.60, and > C0's |
| **C.6** | the skyline is reached | C1/SKY ≤ 1.25 |
| **C.7** | the parametric gap narrows | C1's R13 gap < C0's |

C.1's threshold is deliberately far looser than the development effect
(3.5–5.4×, i.e. a ratio of 0.19–0.29). A confirmation should not require the
development magnitude to replicate; it requires the effect to be real and
material.

C.5 and C.7 are mechanism conditions. A pass on C.1–C.4 with a failure on
C.5 or C.7 would mean the score improved for a reason other than the one the
diagnosis claims, and must be reported as such rather than as a confirmation.

---

## 6. Declared failure branches

- **If C.1 fails**, R11 was a development artifact. It is withdrawn, the
  Phase-2 skyline-gap claim is reinstated, and the diagnosis document is
  corrected.
- **If C.1–C.4 pass but C.5 or C.7 fails**, the improvement is real but the
  variance-collapse explanation is not. Report the effect, withdraw the
  mechanism, and open a fresh diagnosis.
- **If C.6 fails but C.1–C.5 pass**, R11 helps materially without reaching
  the skyline; report the residual gap honestly and do not claim parity.
- **No retuning of R11 under any branch.** The correction is one scalar with
  no free parameter, and adding one would convert a mechanism into a knob.

---

## 7. What Phase 3 cannot conclude

- Nothing about wavelets, scattering, or any fixed compositional geometry:
  that thread is closed and is not re-opened here.
- Nothing about pretrained encoders — none is loaded anywhere.
- Nothing about ImageNet, FID, 32×32 or above, or the paper's trained model.
  The paper trains at a different scale, with a feature encoder, a different
  batch and schedule, and may not exhibit this contraction at all.
- The skyline remains a *distribution-matching* reference at one budget on
  one generator, not a claim about generative quality in general.
- A positive result is scoped to: *CIFAR-10 at 16×16 and 24×24, one
  generator, raw pixel geometry, three fresh seeds, encoder-free throughout.*
