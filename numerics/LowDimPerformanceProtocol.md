# Low-dimensional performance protocol (frozen specification)

*Concrete instantiation of `LowDimPerformanceRoadmap.md` (D0-D5), written
BEFORE the experiments were run.  Code: `lowdim_drift.py` (module),
`lowdim_benchmark.py` (D0-D3 runner), `lowdim_generator.py` (D4).  Runs in
`lowdim_runs/<id>/` with full provenance.  Master seed 20260719.*

All arms use the **exact dimension-general row/column bi-softmax Algorithm-2
estimator** (cross-checked against `driftlab.compute_v_paper`); SNIS appears
only in explicitly labeled diagnostics.  Compute is matched in kernel-pair
evaluations (policy setup costs included).  Oracle mode information is used
only for diagnostics, never by policies.

## Target splits (fixed here, never revised after results)

* **DEV** (code development, smoke): `DEV-2d-K4` (K4 d2 σ0.15 L1 equal).
* **VALIDATION** (baseline tuning D0, ablations D1, policy selection D2):
  `V-1d-K3`, `V-2d-K5`, `V-2d-K4-uneq`, `V-3d-K4`, `V-2d-K4-wide` (σ0.30).
* **HELD-OUT** (D3 final comparison only; frozen policies; no retuning):
  `H-1d-K4-eq`, `H-1d-K5-uneq` (σ0.12), `H-2d-K6-sep` (σ0.10),
  `H-2d-K3-overlap` (σ0.35), `H-2d-K4-hetero` (σ per mode 0.08-0.30),
  `H-ring`, `H-circles`, `H-moons`, `H-skew` (connected, skewed,
  heavy-tailed) — each under `missing` (central cloud, 0.25·scale) and
  `covered` (target sample + 5% jitter) initializations.

Gaussian families are normalized to minimum mode separation `L = 1`.

## Common training configuration

Particles `N = 48`; target minibatch `B = 64`; base horizon 400 updates;
threshold `ED² < 0.05·scale` checked every step, right-censored; reference
samples: 1024 (final), 256 (crossing).  Divergence = non-finite or norm blowup.

## D0 — baseline freezing

Baseline = exact estimator, **eye-mask ON** (the paper's reuse pattern),
fixed bandwidth `τ` and step `η` **tuned on VALIDATION targets** over the
grid `τ ∈ {0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.75}` (normalized units) ×
`η ∈ {0.05, 0.15, 0.4}·τ`, 3 tuning seeds per cell, `missing` init.
Selection score: mean over validation targets of `log(median final ED²)`
(geometric-mean aggregation, prevents any one target dominating).
The literal paper temperature set `{0.02, 0.05, 0.2}` is recorded as a
reference arm (temperature baseline, not "the paper's model").
Tuning cost (kernel pairs) is recorded and the same budget order is
available to the modified method's setup.  Frozen output:
`lowdim_runs/<id>/baseline_frozen.json`.

Gate: invariants pass (translation, finiteness, matched-batch cancellation,
1-d driftlab agreement); tuning cost recorded; no held-out target touched.

## D1 — exact-estimator ablations (validation targets, missing init)

Arms (one factor at a time; 6 paired seeds):

1. `base` — frozen D0 `(τ*, η*, mask on)`;
2. `coarse-only` — `τ = L̂` (k-means estimate from setup sample), `η = η*`;
3. `c2f-only` — geometric `τ_t: L̂ → σ̂` over 70% of updates, `η = η*`
   (bandwidth-only causal comparison; same per-step kernel cost);
4. `step-only` — `τ = τ*`; `η` from the coupled-generator safety rule:
   one-time full central-difference Jacobian of the paper field on a
   16-particle surrogate at `τ*` (cost counted), candidates
   `{0.05, 0.1, 0.25}·η*_ceiling`, plus `{0.1, 0.5}·τ` as cheap references;
5. `mask-only` — mask OFF when `N < 8·K̂` (K̂ from silhouette k-means),
   else ON;
6. `combined` — c2f bandwidth + joint step `η_t = min(0.1·τ_t, cap)` with
   `cap = 0.25·η*_ceiling` + the mask rule.

Gate: mechanisms tested on the exact estimator; bandwidth/step separated;
at least one combined candidate improves the validation aggregate
(geometric-mean ED²), else record non-transfer and redesign.

## D2 — adaptive annealing trigger (validation targets)

Fixed-time 70% annealing is the baseline schedule.  Candidate observable
triggers (start coarse at `τ₀ = L̂`, anneal geometrically to `σ̂` over the
remaining budget once triggered):

* `plateau`: relative change of a moving window of the on-batch mean
  `‖V̂‖` under the coarse bandwidth falls below 15% over 40 updates;
* `agree`: mean cosine between coarse-τ and fine-τ drift estimates on the
  current batch exceeds 0.8 (extra fine evaluation cost counted);
* `fixed70`: the fixed-time baseline schedule.

Selection on validation aggregate at matched compute; the winner must beat
or match `fixed70` after its extra evaluation cost is charged.  Frozen
output: `policy_frozen.json` specifying initial bandwidth estimate,
trigger, bandwidth update, step rule, mask rule, and fallback
(= `fixed70`) when diagnostics are ambiguous.

## D3 — held-out benchmark and gate

Frozen `base` vs frozen `modified` on all HELD-OUT cells
(9 families × 2 inits), **20 paired seeds** each (same init and target
batch stream per seed).  Primary: final ED², KM-median threshold time,
kernel pairs, wall time.  Secondary: sliced W₁ (32 projections), mode
coverage/mass error (Gaussian families only), on-support residual,
divergence counts.

Acceptance gate (declared here, before running):

1. paired aggregate median final-ED² ratio (modified/base) ≤ 0.8;
2. upper end of the paired 95% bootstrap CI of that aggregate ratio < 1;
3. cells degraded by more than 10% must be ≤ 20% of cells;
4. KM threshold time not worse in aggregate at matched compute;
5. the improvement must not vanish when the four Gaussian-mixture-only
   families are excluded (recompute the ratio on the non-Gaussian +
   hetero/overlap cells).

Failure ⟹ return to D2; the gate is not weakened post hoc.

## D4 — learned-generator transfer

Fixed MLP `G_θ: R² → R^d` (2→64→64→d, tanh), Adam(1e-3), latent batch 64,
2000 updates, identical init/latent stream/target stream/optimizer across
arms.  Update: regression of `G(z)` toward `stopgrad(x + η·V̂(x))` with
`x = G(z)` and `V̂` from the frozen policy (base or modified).  Invariants:
matched-batch zero-update; gradient direction agreement with `−JᵀV̂` at
small η.  Compare frozen D0 baseline vs frozen D2 policy on a held-out
subset (`H-1d-K4-eq`, `H-2d-K6-sep`, `H-moons`, `H-skew`), 10 paired seeds,
paired bootstrap CI.  No retuning on held-out targets.

## D5 — consolidation

`LowDimPerformanceResults.md` with the complete algorithm spec, tables,
paired CIs, ablations, failures, wall/tuning costs, scope statement, and
links from `DynamicsRoadmap.md`/`ResearchStatus.md`.

## D1 outcome and pre-D2 redesign addendum (2026-07-19, before D2 ran)

D1 gate result: **no modification arm beat the tuned baseline on the
validation aggregate** (base 0.00543; best modification 0.00624).  Recorded
per protocol as a non-transfer of the SNIS-era headroom: the strong-baseline
principle already absorbed the coarse-bandwidth mechanism (the tuned
`τ* = 0.35 ≈ √(σ·L)`), the mask rule is inert at `N = 48`
(`N ≥ 8·K̂` everywhere), and annealing to `σ̂` overshoots below the
grid-optimal scale.

Redesigned candidate family for D2 (validation-selected, still non-oracle):
the modification's remaining edge is being **tuning-free and
geometry-matched** where the baseline needed a 7.1e8-kernel-pair sweep and
carries one fixed `τ*` across geometries.  D2 therefore selects among:

* `geo-fixed`: `τ = √(σ̂·L̂)`, `η = 0.15·τ`, mask rule unchanged;
* `geo-anneal-fixed70` / `geo-anneal-plateau`: start `τ₀ = L̂`, anneal
  geometrically to `√(σ̂·L̂)` (not `σ̂`), `η_t = 0.15·τ_t`;
* the original `anneal-to-σ̂` triggers (fixed70/plateau/agree) as controls.

The D3 gate is unchanged; D3 tests whether geometry-matching generalizes to
held-out families where the frozen `τ* = 0.35` may be mismatched.
