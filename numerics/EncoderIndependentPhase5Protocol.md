# Encoder-Independent Kernel Drifting — Phase 5 protocol

## Reforms R15–R19: reporting integrity, and the mechanism-targeted candidate

*Frozen pre-outcome design. Every threshold was fixed before any Phase-5 run.
Source evidence: `EncoderIndependentPhase4Diagnosis.md`. Results go to
`EncoderIndependentPhase5Results.md`.*

---

## 0. What Phase 4 left, and one further correction

Phase 4 failed all three gates. The diagnosis explained why, and in doing so
corrected a Phase-4 claim. **This protocol corrects the correction**, because a
further measurement changed it again — recorded here rather than buried:

1. Phase 4 said *"R11 fails at the paper's declared operating point."*
2. The diagnosis said that was mislabeled — the τ values collapse the kernel
   *in raw pixel space*, and the paper's grid is calibrated for normalized
   encoder features.
3. **That is also not right.** τ is normalized by the median pairwise distance,
   so it is geometry-independent by construction. Measured on CIFAR-16, the
   affinity distribution at each τ is **identical** across raw pixels,
   wavelet, scattering and a learned encoder (median affinity 2.06e-09 at
   τ = 0.05 in all four). Normalized distance spread is q05/median ≈ 0.67 in
   every one of them.

So the accurate statement is: **the paper's τ grid puts the kernel in a
near-nearest-neighbour regime at N = 64 for real-image data in any of these
geometries.** That is not a defect of my implementation and not specific to
pixels.

This is **consistent with the repository's own certified analysis.** Experiment
E4 (`numerics/RESULTS.md`) tabulates softmax ESS at N = 64 against normalized
distance spread `s`: at `s = 0.1`, τ = 0.05 already gives ESS 7.6 of 64.
CIFAR-16's spread is `s ≈ 0.20` — double the widest case E4 tabulated — and the
measured ESS is 6.5. E4 predicted this regime; Phase 4 walked into it without
checking.

The consequence for R19 is that it stops being "test in a normalized feature
space" (already done, changes nothing) and becomes a **batch × τ interaction
test**: if the τ failure is neighbour starvation, raising the batch at fixed τ
should recover it.

---

## 1. Reforms

### R15 — kernel-health precondition *(blocking, reporting integrity)*

No cell may be scored as a method result if its kernel is numerically dead.
Before any arm runs in a cell, measure field health and **exclude** the cell
unless:

```
collapsed_row_fraction == 0   and   ess_fraction >= 2/batch
```

The second condition requires each probe to see at least two effective
neighbours; below that the field is a nearest-neighbour rule and the
"kernel geometry" being compared does not exist.

P4A scored three cells in which 94%, 0% and 0% of rows were collapsed with
median affinities of 4e-20, 1e-08 and 1e-02, and drew a conclusion about the
paper from them. This reform makes that impossible rather than unlikely.

Excluded cells are **reported with their health numbers**, not silently
dropped.

### R16 — damped / decaying teacher step *(the substantive candidate)*

The teacher map loses 2–8% of effective dimension per application at constant
variance, worst mid-trajectory. Free particles — which do not collapse — apply
a *decaying fractional* step, `0.2 · (1 − t/T)`; the regression applies its map
at full strength every step.

Implement `ObjectiveConfig.eta_schedule ∈ {"constant", "linear_decay"}`:

```
eta_effective = step_eta                      (constant)
eta_effective = step_eta * (1 - progress)      (linear_decay)
```

This mirrors the free-particle schedule exactly. It targets the measured
mechanism rather than compensating for its symptom, and it has no
scale-matching side effect.

**Compared head-to-head against R11**, and in combination.

### R17 — latent dimension

Two independent diagnostics point here: P4D found the residual skyline gap is
a missing spectral tail (real data carries 13.4% of variance beyond the top 32
directions, the corrected generator 4.7%), and H1 found the generator
underparameterized for point-target regression. Sweep `latent_dim ∈ {32, 64,
128}` at fixed width.

### R18 — measure the contraction on the real trajectory

H2's contraction was measured on a synthetic interpolation. Log, during actual
training, the effective dimension of the generator output and of the teacher,
and their ratio. This confirms or refutes the surviving mechanism hypothesis
as the previous two were refuted.

Implemented as a standing diagnostic (`teacher_dimension_ratio`), reported
every run like ESS and coverage.

### R19 — batch × τ interaction

Cross `τ ∈ {0.05, 0.2, ESS-calibrated}` with `batch ∈ {64, 256, 1024}` and
record health alongside score. If the τ failure is neighbour starvation, ESS
should scale with batch at fixed τ and the score should recover.

Cells failing R15 are excluded from scoring but reported.

---

## 2. Frozen design

| Item | Value |
|---|---|
| Data | CIFAR-10 at 16×16, disjoint `train`/`eval`, no labels |
| Seeds | 3, fresh: `MASTER_SEED + 5000..` |
| Budget | 600 steps |
| Geometry | raw pixel, paper Algorithm-2 field |
| Score | `normalized_geometry_score_v2`, null averaged over 5 draws |

### Arms for the R16-versus-R11 comparison

| ID | R11 variance match | R16 η schedule |
|---|---|---|
| **D0** | off | constant | *(the uncorrected baseline)* |
| **D1** | **on** | constant | *(Phase-3's confirmed R11)* |
| **D2** | off | **linear_decay** | *(R16 alone)* |
| **D3** | **on** | **linear_decay** | *(both)* |

## 3. Exit gate

| ID | Condition | Threshold |
|---|---|---|
| **G5.1** | R16 alone materially beats the baseline | D2/D0 paired ratio ≤ 0.60, bootstrap upper bound < 1 |
| **G5.2** | R16 repairs the collapse | D2 effective-dimension ratio ≥ 0.60 |
| **G5.3** | the mechanism is confirmed on the real trajectory | D0's logged `teacher_dimension_ratio` < 0.98; D2's or D1's strictly greater than D0's |
| **G5.4** | R15 excludes what it should | every cell with `collapsed_row_fraction > 0` is excluded, and ≥ 1 such cell exists in the R19 grid |
| **G5.5** | the batch × τ interaction is as predicted | at fixed τ = 0.05, `ess_fraction` rises monotonically with batch |

G5.1 and G5.2 decide whether R16 supersedes R11. **G5.3 is the mechanism
condition** — a score improvement without it means the effect is real and the
explanation is still wrong, which must be reported as such.

R17 and R19 are **reported, not gated**: they are scoping sweeps, and a
threshold on them would be invented rather than derived.

## 4. Declared failure branches

- **G5.1 fails** → damping is not the lever; R11 stands as the only working
  repair and the mechanism account weakens further.
- **G5.1 passes, G5.3 fails** → report the effect and withdraw the mechanism.
  This would be the *third* refuted mechanism; at that point the honest move
  is to stop proposing them and report the phenomenology.
- **G5.3 passes but D2 does not beat D1** → both work; prefer R16 on
  parsimony (it has no scale-matching side effect) and say so explicitly.
- **No retuning.** `eta_schedule` is a declared choice between two options,
  not a continuous knob to search.

## 5. What Phase 5 cannot conclude

- Nothing about fixed compositional geometry (closed, three negatives).
- Nothing about the paper's results. Even a clean pass says *this transfer
  step contracts and here is a better step rule* — the paper trains with an
  encoder, CFG and a self-mask at a scale this does not touch.
- Nothing about ImageNet or FID.
- A positive result is scoped to: *CIFAR-10 at 16×16, raw pixel geometry, one
  generator family, three fresh seeds, 600 steps.*
