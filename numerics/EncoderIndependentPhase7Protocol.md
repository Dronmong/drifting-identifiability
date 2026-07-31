# Encoder-Independent Kernel Drifting — Phase 7 protocol

## Ask of the kernel what Phase 6A asked of the optimizer

*Frozen pre-outcome design. Every threshold below was fixed before any
Phase-7 run. Source: `EncoderIndependentPhase6Followup.md`. Results go to
`EncoderIndependentPhase7Results.md`.*

---

## 1. Why this phase exists

Phase 6A tested whether R11 was compensating for a mis-set **optimizer** and
answered no, cleanly: 0 of 12 cells reached the second-moment band across a
60× learning-rate range and three optimizers.

The Phase-6 follow-up then found a candidate with far better support. Across
every R15-admissible bandwidth at CIFAR-16, the free-particle fixed point and
its quality are **monotone in the kernel's realized neighbour count, with no
crossings** — from ESS 0.146 (second moment 0.304, ED² 1.669) to ESS 0.978
(0.856, ED² **0.0708**). The program has used ESS 0.5 since Phase 2, which
sits near the bad end and costs a factor of ~5. Free particles at τ = 1.0
already **beat the R11-corrected generator**, 0.071 against 0.160.

So R11 has never been tested against a properly set kernel, and the deficit
it corrects is largely a bandwidth artifact. A second axis compounds it: the
field's **cloud size**, where 64 samples — the generator's training batch —
costs about an order of magnitude against 512 in every arm.

**This protocol can retire R11**, exactly as Phase 6's could. Free particles
already clear both of the gate conditions below at τ = 1.0, so the gate has a
real chance of firing, which is precisely why it is frozen here.

---

## 2. Phase 7A — sweep the kernel for the generator

| axis | values |
|---|---|
| bandwidth | ess 0.5 *(incumbent)*, ess 0.9, τ = 1.0, τ = 2.0 |
| field cloud (R27) | 64 *(incumbent)*, 256, 512 |
| R11 | off, on |

Data: CIFAR-10 at 16×16, raw pixel geometry, paper Algorithm-2 field.
Seeds: 3 fresh, `MASTER_SEED + 12000..`. Budget 600 steps, target batch 64.
**Optimizer held at Adam / 2e-3** — 6A established it does not matter, and
holding it fixed keeps this a one-axis test.

**R15 admissibility is evaluated first.** A cell whose kernel is numerically
dead is not evidence about kernel geometry; the follow-up's τ = 0.02 arm
posted the best second moment in its whole sweep while being 100% collapsed
with median affinity 1.9e-23. Inadmissible cells are **reported with their
health numbers and not scored**.

### 7A gate — the decisive one

> **If any *admissible* (bandwidth, cloud) cell *without* R11 reaches a
> second-moment ratio inside `[0.7, 1.3]` and an ED² within 25% of the best
> R11 cell, then R11 is superseded by setting the kernel correctly.**

If that fires, R11 is reinterpreted as having compensated for a mis-set
bandwidth and cloud size, and Phases 3, 5 and 6 are re-scoped as having
measured a bandwidth artifact at one operating point.

---

## 3. Phase 7B — the particle/generator gap at the good operating point

At ESS 0.5 the corrected generator (0.160) beat free particles (0.349). At
τ = 1.0 free particles (0.071) beat the corrected generator. **The ordering
inverts**, which would mean the generator — not the field — becomes the
bottleneck once the kernel is set properly.

7B runs particles and the generator at matched bandwidth and matched cloud
size, across the same bandwidth axis, and reports the gap. Reported, not
gated: it fixes the wording of any claim this program makes about what the
amortization costs.

---

## 4. Phase 7C — is there a bandwidth *rule*, or only a lucky value?

τ = 1.0 was the follow-up's largest tested value and quality was **still
improving**, so the sweep established a direction, not an optimum.

- extend the particle sweep to τ ∈ {1, 2, 4, 8} to locate the turn;
- sweep the ESS target over {0.5, 0.7, 0.9, 0.95, 0.99};
- for every arm, record the **target-only** realized ESS alongside quality.

The question is whether the good setting is predictable from target data
alone. The ESS-targeting rule was built to be exactly that and it chose 0.5,
which the follow-up shows is poor. **A rule that picks the right bandwidth
from target data alone is worth more than the best hand-chosen τ**, because
it is the part that transfers. Reported, not gated — a threshold on it would
be invented rather than derived.

---

## 5. Reform landing with this phase

- **R27** — `TrainConfig.field_cloud`: how many generator samples form the
  field's cloud, previously tied silently to the target batch. `None` keeps
  the old behaviour exactly.

---

## 6. Declared failure branches

- **7A finds a clean admissible no-R11 cell** → R11 is superseded; re-scope
  Phases 3/5/6 and say so plainly.
- **7A finds none** → R11 survives a second sharp test, now against the
  best-supported alternative explanation remaining.
- **The best cells are all inadmissible** → report that the quality gain
  found in the follow-up lives where the kernel is degenerate, and treat the
  follow-up's headline as scoped accordingly.
- **7C finds no target-only rule** → report τ = 1–2 as a hand-chosen setting
  with no derivation, which is weaker and must be said.
- **No tuning.** Both grids are declared above and are swept in full, not
  searched for a cell to headline.

## 7. What Phase 7 cannot conclude

- Nothing about ImageNet, FID, or the paper's trained model.
- Nothing about the paper's full protocol (no CFG, no encoder).
- The geometry thread stays closed; the anchor is not enabled in any arm.
- A positive result is scoped to: *CIFAR-10 at 16×16, raw pixel geometry,
  one generator family, 3 fresh seeds, 600 steps.*
