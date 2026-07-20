# Low-dimensional performance study: results (2026-07-19)

*Final report for `LowDimPerformanceRoadmap.md` D0–D3 (+D4 wiring).
Protocol frozen in advance: `LowDimPerformanceProtocol.md` (with its dated
pre-D2 redesign addendum).  Code: `lowdim_drift.py`, `lowdim_benchmark.py`,
`lowdim_generator.py`.  All runs with manifests in `lowdim_runs/`.
Everything here is measured synthetic evidence; nothing is Lean-certified;
nothing benchmarks the paper's trained image models.*

## Verdict up front

**The pre-declared D3 gate FAILED, and the gate stands.**  Under a strong
(tuned) baseline, matched compute, the exact paper estimator, and held-out
targets, the theory-derived modified procedure does **not** outperform the
base procedure in aggregate (paired ratio 1.031, 95% CI [1.019, 1.041]).

The defensible claims that survive are narrower and still useful:

1. **Tuning-free parity.**  The frozen modified policy — estimate `(σ̂, L̂)`
   from 256 unlabeled target samples, set `τ = √(σ̂·L̂)`, `η = 0.15·τ` —
   matches the grid-tuned baseline on validation (0.00544 vs 0.00529
   geo-mean ED²) at **zero tuning cost**, versus the baseline's 7.1×10⁸
   kernel-pair sweep.
2. **Conditional wins where geometry-matching matters.**  On held-out
   families where the frozen tuned bandwidth is mismatched, the modified
   policy wins large: two-moons **−47%/−41%**, ring **−37%/−21%**, 1-d K4
   **−14%** — while losing 3–10% on the families the baseline was
   effectively tuned for.  Only 1/18 cells degraded by more than 10%.
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
3 seeds, 5 validation targets): frozen baseline `(τ*=0.35, η*=0.0525,
mask on)`, geo-mean ED² 0.00529 — **4.4× better** than the literal paper
temperature set {0.02, 0.05, 0.2} (0.02352).  Tuning cost recorded:
7.096×10⁸ kernel pairs.

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

Gate: crit1 (ratio ≤ 0.8) **FAIL** (1.031); crit2 (CI hi < 1) **FAIL**
([1.019, 1.041]); crit3 (≤20% cells degraded >10%) pass (1/18);
crit4 (threshold time) pass (KM tie); crit5 (non-Gaussian robustness)
**FAIL** (1.046 [1.035, 1.059]).

Why not another D2 loop: D1 bounds every candidate in the tested
bandwidth/step/mask mechanism family within ±25% of the tuned baseline on
validation — no validation-selectable candidate can plausibly reach the
20% aggregate-win criterion, and a redesign aimed at the specific held-out
weaknesses (k-means mis-scaling on concentric circles; no-cluster targets)
would be oracle leakage after seeing the test.  The loop is closed with
the negative result.

### D4 — wiring verified, not run as a decisive test

Per the roadmap ordering, D4 was gated on D3 passing.  The runner
(`lowdim_generator.py`: fixed 2→64→64→d tanh MLP, Adam, drifting
regression update) passes its invariants (gradcheck; matched-batch zero
update), and a labeled smoke wiring check shows the same parity picture
(aggregate 0.998 [0.959, 1.030]).  No performance claim is made from it.

## What this buys the research program

1. **A calibrated negative with a strong baseline** — rarer and more
   useful than the earlier optimistic pass: *for low-dimensional Gaussian
   mixtures, a tuned fixed-bandwidth Algorithm-2 baseline is already at the
   mechanism ceiling; bandwidth/step/mask policies motivated by the
   population theory do not beat it, they only remove its tuning cost.*
2. **A precise, cheap, tuning-free default** with quantified conditional
   value: `τ = √(σ̂·L̂)`, `η = 0.15·τ` — parity on mixtures, 20–47% better
   on curved/mismatched families, no sweep.
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
decisive learned-generator transfer.
