# Encoder-Independent Kernel Drifting — Phase 10 protocol

## Measure the shape law, then act on shape

*Frozen pre-outcome design. Thresholds fixed before any Phase-10 run.
Source: `EncoderIndependentDeficitMechanism.md`. Results go to
`EncoderIndependentPhase10Results.md`.*

---

## 1. Why this phase exists

The mechanism pass established that the deficit is a **shape** phenomenon: at
matched second moment the generator's cloud is radially balanced (+0.0006)
while a data-shaped cloud at the same scale is pushed outward 43× harder
(+0.0264). The generator's cloud is clumped — nearest-neighbour spacing 3.47
against 6.36 at equal variance — because it concentrates energy in few
directions (spectral tail 0.0034 against 0.142).

Two things follow, and this phase does both.

**It is not yet a law.** The relationship is demonstrated at two scales, not
quantified. And **spacing and spectral tail are correlated in every cloud
measured so far**, so the pass could not say which one the field responds to.
They imply different interventions, so 10A must separate them.

---

## 2. Phase 10A — the law, with the two candidates separated

For each constructed cloud, scan the scale and locate the radial zero — the
second moment at which that cloud shape is in equilibrium.

**Family S (spectrum).** Reweight the data's singular values as `S^β`,
renormalized to fixed total variance. `β > 1` concentrates the spectrum
(lower tail), `β < 1` flattens it (higher tail). Sweep
`β ∈ {0, 0.5, 1, 1.5, 2, 3}`. This moves the tail **and** spacing together.

**Family P (packing).** Hold the covariance **exactly fixed** — same spectrum,
same eigenbasis — and vary only the regularity of the point arrangement, by
applying repulsion relaxation and rescaling each principal coordinate back to
its target variance after every step. Sweep the relaxation strength. This
moves spacing at **fixed tail**.

### Declared reading

- radial zero tracks the tail in S **and is flat in P** → **the spectrum** is
  the variable; the intervention should be a tail floor;
- radial zero tracks spacing in **both** → **packing** is the variable; the
  intervention should be a repulsion term;
- both matter → report both slopes and let 10B test both.

**Also declared:** the law is scored against three points already recorded,
which were not used to build it — generator tail 0.0034 → second moment 0.42,
R11 0.047 → 0.95, free particles 0.415 → 1.0. If the fitted relation does not
order these correctly it does not describe the recipe, and 10B proceeds on
the measurement alone.

Reported, not gated.

---

## 3. Phase 10B — intervene on shape

| arm | intervention |
|---|---|
| E0 | none *(baseline)* |
| E1 | R11 scalar second-moment match *(incumbent)* |
| E2 | **repulsion penalty** on the generator's own batch |
| E3 | **spectral-tail floor** on the generator's own batch |
| E2+E3 | both |

Weights are a **declared sweep, not a search**: `λ ∈ {0.1, 1.0}` for each,
every cell reported. Bandwidth held at target ESS 0.9, field cloud 256,
Adam/2e-3, CIFAR-16, 3 fresh seeds (`MASTER_SEED + 18000..`), 600 steps.

Neither intervention touches the cloud's scale. E2 minimizes
`mean_{i≠j} exp(−‖xᵢ−xⱼ‖/h)`, which pushes points apart without reference to
any target scale. E3 maximizes the fraction of batch variance beyond the top
32 directions. **If the mechanism is right, opening the shape should fix the
scale as a consequence** — that is the phase's real claim.

### 10B gate

> **If a shape intervention without R11 reaches a second-moment ratio inside
> `[0.7, 1.3]` and an ED² within 25% of the best R11 cell, then R11 is
> superseded by acting on shape** — and the program's long-standing empirical
> reform is replaced by one derived from a measured mechanism.

Same shape as the gates of Phases 6, 7 and 8, which R11 survived three times.

---

## 4. Declared failure branches

- **10B gate fires** → R11 superseded; re-scope and report the shape
  intervention as the result.
- **Shape interventions move the shape but not the second moment** → the
  mechanism is **refuted**: shape would then be a correlate, not a cause.
  This is the outcome that would overturn the mechanism pass, and it is the
  reason the arms log the realized tail and spacing, not just the score.
- **Shape interventions do not move the shape** → the penalties are too weak
  at the declared weights; report as inconclusive with the realized shape
  statistics, and do not read it as evidence about the mechanism.
- **No tuning.** Both grids are declared above and swept in full.

## 5. What Phase 10 cannot conclude

- Nothing about ImageNet, FID, or the paper's trained model.
- Family P holds the covariance fixed only up to the per-coordinate rescaling
  described; higher moments are free to change and are not controlled.
- A positive result is scoped to CIFAR-16, raw pixel geometry, one generator
  family, target ESS 0.9, 3 fresh seeds, 600 steps.
- The anchor stays disabled; the geometry thread stays closed.
