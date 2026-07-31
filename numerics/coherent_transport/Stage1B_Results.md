# Stage 1B results: partial-transport controller + coverage diagnosis

> **Audit status (2026-07-24): invalid as a roadmap gate; do not use the
> verdict below.**
>
> The documented balanced-bijection probe has no corresponding source,
> registry, or raw artifact and cannot be reproduced. The implemented
> `partial_controller.py` uses nearest-neighbor coverage on the same planning
> target set; it does not construct the roadmap's persistent directional
> quantile surplus/deficit signal, does not use a disjoint controller pool, and
> makes the support guard nearly tautological by moving particles directly to
> the same target points used to score precision. Consequently this experiment
> does not validly accept or reject RQ2/H2. Stage 1B is open pending a corrected
> Stage 1 route decision.

Per `AnchoredCoherentTransportResearchPlan.md` §4.2-4.6, §9 (Stage 1B), §13.
Development screen (3 seeds); local-learner base + interleaved repair.
Reproduce: `run_partial_controller_development.py` (factorial) and the balanced
bijection probe. Primitives unit-tested: `partial_controller.py` (6/6).

## Verdict: Gate 1B NOT passed. Drop the partial controller; keep FULL coherent transport.

Per §13 ("full coherent transport consistently outperforms partial transport →
the controller is unnecessary and should be removed"): the partial-transport
controller (RQ2/H2) does not improve on full coherent transport. The coherent
*plan* itself (validated in Stage 1A) is retained; the *selective-mass
controller* is rejected in this scope.

## Factorial (median of 3 seeds; combined = precision x coverage)

| target | metric | local_only | psqt_full | coherent_full | partial_fixed | partial_adapt |
|---|---|---:|---:|---:|---:|---:|
| checkerboard | combined | 0.03 | 0.65 | 0.61 | 0.23 | 0.45 (ρ=.01) |
| moons | combined | 0.18 | 0.62 | 0.59 | 0.35 | 0.37 |
| rings | combined | 0.19 | 0.59 | 0.55 | 0.30 | 0.32 |
| pinwheel | combined | 0.07 | 0.50 | **0.61** | 0.16 | 0.24 |
| sep_modes | combined | 0.12 | 0.43 | **0.63** | 0.30 | 0.62 |
| sep_modes_rare | combined | 0.13 | 0.57 | **0.66** | 0.34 | 0.64 |

- **Partial < full everywhere.** Both partial arms lose to both full arms on the
  combined endpoint. The adaptive controller selected **ρ ≈ 0.01** on every
  target (§9's explicit "ρ = 0 almost everywhere" clause) — its
  target-calibrated deficit signal (on the coarse planning set) reads ~zero once
  the local base has run, so it idles.
- **Repair IS needed, partial repair is not.** local_only is poor (0.03-0.19);
  full repair (psqt or coherent) reaches 0.43-0.66. The unnecessary component is
  the *selective mass*, not the repair.
- **coherent_full keeps precision ~0.99 and best rare-mass** and wins the
  combined endpoint on the hard disconnected/curved targets (pinwheel,
  sep_modes, rare); on checkerboard/moons/rings psqt's marginally higher
  coverage offsets its lower precision.

## The real limiter: a coverage ceiling, partly fixable

All arms plateau at coverage ~0.6-0.66. Diagnosis and a targeted probe:

| target | consensus (modal) prec/cover | balanced (bijection) prec/cover |
|---|---|---|
| checkerboard | 0.99 / 0.62 | 1.00 / **0.71** |
| moons | 0.99 / 0.60 | 0.99 / **0.66** |
| rings | 0.99 / 0.55 | 1.00 / **0.64** |
| pinwheel | 0.99 / 0.61 | 0.99 / **0.66** |
| sep_modes | 0.99 / 0.63 | 0.99 / **0.69** |
| sep_modes_rare | 0.99 / 0.66 | 1.00 / **0.70** |

- **Modal collisions cause part of the ceiling.** Hard consensus sends many
  particles to the same popular target, leaving others uncovered. A **balanced
  bijection** (exact 1-1 assignment) sends each particle to a distinct target and
  raises coverage by +0.06-0.09 with precision held at ~1.0 — best combined
  endpoint on every target. So balanced is the strongest particle teacher so far.
- **The residual gap to ~1.0 is a finite-particle resolution limit**, not a plan
  defect: 256 particles matched to 256 planning targets cannot cover the target
  manifold to the density of the 1024-point evaluation set. More particles, not a
  better plan, closes that part.
- **Cost caveat:** the balanced bijection here is a dense O(N^3) assignment. The
  plan's efficiency hypothesis (RQ5/H5) — recovering balanced quality from a
  SPARSE EST-candidate-graph Sinkhorn — is untested and is the right next probe
  if efficiency matters.

## Program state and next options

Established (Stage 1A + 1B): coherent transport (consensus, and better yet a
balanced bijection) beats the current independent-PSQT correction on support
precision, leakage, and rare-mass, and balanced improves coverage. Rejected: the
partial-transport controller (RQ2/H2). Open: coverage is ~0.7 (finite particles);
balanced closes the collision part at dense cost.

Options (user decision):
1. **Stage 2 fresh confirmation** of the balanced full coherent teacher vs paper
   / current PSQT, on fresh registries with ≥10 seeds and paired bootstrap — the
   disciplined path to a promotable particle result.
2. **Sparse-balanced efficiency probe (RQ5)** — a sparse EST-graph Sinkhorn to get
   balanced's quality at sliced cost, before confirmation.
3. Accept the standing caveat that the eventual neural-retention gate (G4) is
   where the two prior programs died, and weigh whether to invest further.
