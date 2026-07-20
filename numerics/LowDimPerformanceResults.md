# Low-dimensional performance study: results (2026-07-19)

*Final report for `LowDimPerformanceRoadmap.md` D0–D3 (+D4 wiring).
Protocol frozen in advance: `LowDimPerformanceProtocol.md` (with its dated
pre-D2 redesign addendum).  Code: `lowdim_drift.py`, `lowdim_benchmark.py`,
`lowdim_generator.py`.  All runs with manifests in `lowdim_runs/`.
Everything here is measured synthetic evidence; nothing is Lean-certified;
nothing benchmarks the paper's trained image models.*

## Verdict up front

**The pre-declared D3 gate FAILED, and the gate stands.**  Under a tuned
Gaussian-validation baseline, matched per-update kernel cost, the exact paper
estimator, and the fixed held-out suite, the theory-derived modified procedure
does **not** outperform the base procedure in aggregate (paired ratio 1.031).
The originally reported row-bootstrap interval `[1.019, 1.041]` conditions on
the 18 chosen target cells and is too narrow for target-family generalization;
a hierarchical cell-and-seed reanalysis gives approximately `[0.990, 1.062]`.
This statistical correction does not rescue the gate: the point estimate is
still far above the pre-declared `0.8` requirement.

The defensible claims that survive are narrower and still useful:

1. **No-per-target-sweep parity.**  The frozen modified policy — estimate
   `(σ̂, L̂)` from 256 unlabeled target samples, set `τ = √(σ̂·L̂)`,
   `η = 0.15·τ` — approximately matches the grid-tuned baseline on validation
   (0.00544 vs 0.00529 geo-mean ED²) without a new grid search for each target.
   It is not a zero-tuning-cost discovery: D2 selected this rule from six
   candidates after the D0/D1 development runs, and k-means setup has nonzero
   cost.
2. **Exploratory conditional wins by the combined policy.**  The frozen policy
   wins large on two moons (**−47%/−41%**), ring (**−37%/−21%**), and 1-d K4
   (**−14%**), while losing 3–10% on several other families.  These wins cannot
   yet be attributed to geometry matching: the policy simultaneously changes
   bandwidth, step size, and a sample-dependent mask rule.  On the recorded
   seeds the mask was on in only 9/20 ring setups and 6/20 moon setups, versus
   always on for the baseline.  Only 1/18 cell-median ratios exceeded 1.10.
3. **The mechanisms are real but already absorbed by tuning.**  D1's
   one-factor ablations on the exact estimator found **no** modification
   (coarse, coarse-to-fine, generator-derived step, mask rule, combined)
   that beats the tuned baseline on validation — because the D0 grid search
   itself lands on `τ* = 0.35 ≈ √(σ·L)`: a well-tuned fixed bandwidth *is*
   the coarse-bandwidth design rule.

## Stage-by-stage record

### D0 — the baseline is strong and verified

Invariants (translation, finiteness, matched-batch cancellation, exact 1-d
agreement with `driftlab.compute_v_paper`): all PASS.  Grid (7 τ × 3 η,
3 seeds, 5 Gaussian validation targets): frozen baseline
`(τ*=0.35, η*=0.0525, mask on)`, geo-mean ED² 0.00529.  The old 0.02352
“paper-temperature” number came from *cycling* through `{0.02,0.05,0.2}` on
successive updates, not from three fixed-temperature runs or a simultaneous
multi-scale field, so the former “4.4× better than the paper set” comparison is
withdrawn.  The recorded 7.096×10⁸ kernel pairs include both grid selection
and that reference arm.

### D1 — honest non-transfer (validation, exact estimator, 6 paired seeds)

| arm | geo-mean ED² |
|---|---:|
| **base (tuned)** | **0.00543** |
| mask-only (inert at N=48) | 0.00543 |
| step-taumult-0.1 | 0.00624 |
| c2f-only | 0.00629 |
| step-taumult-0.5 | 0.00634 |
| combined | 0.00643 |
| step-ceil-0.05 / 0.1 / 0.25 | 0.00669 / 0.00803 / 0.01040 |
| coarse-only | 0.01310 |

Recorded per protocol: the SNIS-era headroom does not transfer against a
strong baseline on these targets.  Redesign (dated addendum, before D2):
target the *tuning-free / geometry-matched* advantage instead.

### D2 — frozen modified policy

Six validation-selected candidates; winner **`geo-fixed`**
(`τ = √(σ̂·L̂)`, `η = 0.15·τ`, mask on iff `N ≥ 8·K̂`, all quantities from
unlabeled samples), aggregate 0.00544.  Annealing variants were all worse
(0.0061–0.0076): with a well-chosen fixed scale available, spending budget
at other scales only costs.

### D3 — held-out gate (18 cells × 20 paired seeds, frozen policies)

Per-cell median ED² ratios (modified/base): wins — moons 0.527/0.590,
ring 0.628/0.792, 1-d K4 0.857/0.858; parity — K5-uneq, K6-sep; losses —
overlap 1.103/1.030, hetero 1.099/1.055, circles 1.071/1.046, skew
1.055/1.088.

Gate: crit1 (ratio ≤ 0.8) **FAIL** (1.031); crit2 (CI hi < 1) **FAIL** under
both the original row bootstrap (`[1.019,1.041]`) and the target-aware
hierarchical reanalysis (`[0.990,1.062]`); crit3 (≤20% cells degraded >10%)
passes (1/18); crit4 passes, although the aggregate KM tie at step 1 is
dominated by already-covered starts (missing-only KM medians: base 51,
modified 47); crit5 (non-Gaussian robustness) **FAILS** at point ratio 1.046.

Why not tune again on these held-out cells: none of the D1 candidates beats
the baseline, but they are **not** all within ±25%—coarse-only is 141% worse
and the `0.25 η*` arm is 91% worse.  The correct conclusion is that no member
of the tested family provided validation evidence for a 20% aggregate gain.
A redesign targeted to the observed held-out weaknesses would leak test
information.  Any attribution follow-up therefore needs fresh validation and
test targets.

### D4 — wiring verified, not run as a decisive test

Per the roadmap ordering, D4 was gated on D3 passing.  The runner
(`lowdim_generator.py`: fixed 2→64→64→d tanh MLP, Adam, drifting
regression update) passes its invariants (gradcheck; matched-batch zero
update), and a labeled smoke wiring check shows the same parity picture
(aggregate 0.998 [0.959, 1.030]).  No performance claim is made from it.

## What this buys the research program

1. **A calibrated negative against the tested baseline.**  For the chosen
   low-dimensional validation suite, none of the tested
   bandwidth/step/mask policies beats the tuned fixed-bandwidth Algorithm-2
   baseline.  This is evidence about a finite candidate family, not a proof
   that the baseline is a “mechanism ceiling.”
2. **A precise no-per-target-grid heuristic worth attributing.**
   `τ = √(σ̂·L̂)`, `η = 0.15·τ` is near parity on the Gaussian validation
   suite and the combined policy has 20–47% exploratory wins on some curved
   cells.  A fresh factorial experiment must separate bandwidth, step, and
   mask effects before assigning a mechanism.
3. **A sharpened theory target for B1.**  The tuned optimum sitting at the
   *geometric mean* of the two scales is exactly what a residual-floor
   trade-off predicts: transport across separation `L` degrades like
   `e^{−L/τ}` (metastability floor) while local bias grows with `τ`.
   B1 should aim to derive the `√(σL)` scaling of the optimal bandwidth
   from the certified residual bounds — a theory question generated, and
   now precisely constrained, by this study.

## Scope

Synthetic 1-d–3-d targets, empirical-particle model, exact bi-softmax
estimator, matched kernel-pair compute, N=48 particles, 400-update horizon.
Not tested: learned encoder features, images, large N, long horizons,
decisive learned-generator transfer.  The historical D0–D3 artifacts omit
saved trajectories, realized geometry/mask decisions, per-arm wall time,
on-support residual, and complete runner-source snapshots; the fresh
attribution follow-up repairs those auditability gaps rather than rewriting
the historical run.
