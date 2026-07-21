# Mode-recovery program: roadmap

> **OUTCOME (2026-07-20): CLOSED as a characterized negative.** M0 (metric
> library) and M1 (headroom + candidate screen) were executed. Genuine headroom
> exists only at K≥32 or d≥5, but no candidate — additive two-scale (D2) or
> coarse-to-fine annealing (D3) — breaks the resolution barrier (best ≈0.34), and
> the barrier is dynamical, not capacity-bound (D4). The M3 gate / M4 transfer /
> M5 theorem were therefore NOT built. See `M1_headroom.md` and the ledger
> `docs/04-empirical-experimentation/PaperImprovementAttempts.md` §D. The plan
> below is retained as the design record.

Status: proposed next empirical target, 2026-07-20. Supersedes aggregate-ED2
chasing as the primary objective. Motivated by the cross-program pattern in
[`docs/04-empirical-experimentation/PaperImprovementAttempts.md`](../docs/04-empirical-experimentation/PaperImprovementAttempts.md):
across every prior program, the one axis with real, repeated, non-conditional
wins was **missing-mode / mode-coverage recovery**, while aggregate error on
easy, well-tuned, low-dimensional targets had no headroom.

## Objective

> Establish a **general improvement in mode coverage and missing-mode recovery**
> over a strong, tuned exact-paper baseline, in a regime where a single fixed
> bandwidth provably cannot both *reach* distant modes and *resolve* nearby
> ones, under matched architecture and compute, on fresh pre-registered target
> families — and determine whether that improvement transfers to a
> learned generator trained with the paper's exact (Adam) objective.

"General" = target-balanced across mode counts, dimensions, and adversarial
initializations. The primary metric is **coverage**, not aggregate distributional
error. Aggregate error is retained only as a secondary "did we blur the target"
guard.

## Design invariants (the ledger's lessons, made binding)

These are non-negotiable; every prior failure violated one of them implicitly.

- **L1 — Change the scoreboard.** Primary metric is mode coverage / time-to-cover
  under adversarial init, not aggregate ED2. We kept winning coverage and then
  averaging it away.
- **L2 — Verify headroom first.** Before building any candidate, prove the tuned
  paper baseline has a *real coverage deficit* in the chosen regime. If the
  baseline already covers everything, there is nothing to beat (the low-dim
  study's core mistake).
- **L3 — No oracle in decisions.** True mode locations / counts / weights are used
  only to *evaluate* coverage, never inside the field or any policy switch.
- **L4 — Fixed multi-scale over triggered switching.** Prefer a field with both
  scales always active to a geometry-triggered switch; every triggered rule so
  far misfired (A6, B4). A trigger is a later ablation, not v1.
- **L5 — Cheap optimizer diagnostic before the expensive gate.** Run the
  SGD-vs-Adam probe on the candidate *early*. The NCJ win was optimizer-conditional
  and we only found out after a full generator gate; never pay that twice.
- **L6 — Directional, not magnitude.** The candidate must help by changing *which
  modes the field can see* (a directional/reach signal Adam cannot reconstruct),
  not by changing field magnitude (which Adam normalizes away — the reason NCJ
  did not transfer). State this as the falsifiable transfer hypothesis.
- **L7 — Standard discipline.** Fresh disjoint validation/test registries with
  hashes; matched compute (two kernels ⇒ report doubled kernel-pair budget and
  give the baseline the same budget); pre-registered gate never weakened post
  hoc; honest stop rules; clean tree for scientific runs.

## Why this can transfer where NCJ did not

NCJ's no-freeze win was a *magnitude* effect (move faster where the signal is
weak); Adam supplies that independently, so it did not transfer. The coarse-scale
reach effect is a *directional* effect: a large-bandwidth Laplace/Gaussian field
has non-negligible pull toward a mode at distance `L` (its influence decays only
polynomially/slowly in the companion potential), whereas a fine field's pull
decays like `e^{-L/τ}` and points nowhere useful — the cloud literally cannot
feel a distant uncovered mode. No per-parameter gradient normalization can
manufacture a direction that is absent from the field. This is the specific,
pre-registered reason to expect transfer; M4 tests it, and M5 aims to certify it.

## Phases

### M0 — Metric and instrumentation
- Define, in a new `mode_recovery.py`, oracle-*evaluated* metrics:
  - **coverage(q)** = `(1/K) Σ_k 1[ #{particles within ρ·σ_k of mode m_k} ≥ c_cov ]`
    (and a mass-weighted variant `Σ_k w_k · (...)`);
  - **time-to-cover** = first step coverage ≥ `θ` (e.g. all modes), right-censored;
  - **mode-drop rate**, and **post-coverage mass calibration** (are covered modes
    weighted correctly, to catch "cover but wrong mass").
- Invariant tests: a cloud drawn from all modes scores coverage 1; a cloud at one
  mode scores `1/K`; translation/relabel invariance; ρ and `c_cov` documented and
  frozen. Oracle values enter metrics only, asserted unreachable from the field.
- Deliverable: `mode_recovery.py` + tests, committed.

### M1 — Construct the "one bandwidth cannot win" regime + headroom precondition
- Fresh validation (≥12 cfgs) and test (≥16 cfgs) registries, disjoint from all
  prior, with large scale separation `L/σ` (target ≳ 30) and stress on
  `K ∈ {8, 16, 32}` modes and `d ∈ {2, 5, 10}`. Adversarial inits: missing (start
  in one mode), far, concentrated.
- **Precondition (L2):** demonstrate, before any candidate exists, that the
  best fixed-bandwidth paper baseline (swept over τ) has a real coverage deficit
  under missing-mode init here — i.e. no single τ both reaches and resolves.
  Show the coverage-vs-τ curve is non-overlapping between "reach" and "resolve."
  If the baseline already covers, the regime is too easy → redesign M1, do not
  proceed.
- Deliverable: frozen registries + a `M1_headroom.md` recording the precondition.

### M2 — Candidate multi-scale field
- v1: **fixed additive two-scale field** `V = V(τ_coarse) + V(τ_fine)`, both
  scales always on. `τ_coarse`, `τ_fine` set from *unlabeled* data-spread
  quantiles (L3), not from true modes. This mirrors the atlas two-bandwidth
  rescue (the biggest historical win, A3) and avoids the dimension-dependent
  annealing of A2 and the misfire of triggered rules (L4).
- Reuse the audited affinity in `identifiability_drift.py`; add the two-scale
  field with diagnostics (per-scale field norms, coarse/fine disagreement angle,
  per-scale ESS). Invariant tests incl. matched-batch cancellation per scale and
  `τ_coarse=τ_fine` reducing to the single-scale field.
- Compute honesty (L7): two kernels ⇒ ~2× kernel pairs; define a compute-matched
  paper baseline (larger batch or extra update) fixed before testing.
- Later ablations (not v1): coarse-to-fine anneal; a *diagnostic-triggered*
  scale gate using only observable signals (coarse/fine disagreement angle,
  fine-scale ESS collapse) — never oracle mode count.

### M3 — Particle mode-recovery gate
- Pre-registered gate on the untouched test registry, hierarchical bootstrap over
  target cells then seeds. Primary criteria (thresholds fixed here, pre-test):
  1. target-balanced coverage ratio (paper−candidate)/... improvement, e.g.
     mean coverage uplift `≥ +0.15` absolute or time-to-cover ratio `≤ 0.7`;
  2. hierarchical 95% CI excludes zero uplift;
  3. ≥60% of cells improve coverage;
  4. no target family degrades coverage;
  5. **secondary guard:** aggregate ED2 not worse by more than a small
     pre-declared tolerance (do not trade coverage for blur — the jitter failure
     C3);
  6. matched-compute baseline also beaten.
- Honest stop rules: if only some `K`/`d`/init cells improve → conditional design
  rule, not a general win; if the secondary guard fails → we bought coverage with
  blur, reject.

### M4 — Optimizer diagnostic THEN generator transfer
- **First, cheap (L5):** run the SGD-vs-Adam probe on the frozen candidate on a
  few M1 cells. Confirm the coverage benefit is present under **Adam** (the
  directional hypothesis L6). If it vanishes under Adam like NCJ's did, stop and
  record the negative *before* building the full generator gate.
- If it survives Adam: freeze a generator protocol (exact paper MLP/Adam/
  stop-gradient semantics; only the field construction differs) and run the
  transfer gate on a fresh generator split, coverage-primary.

### M5 — Certified layer (mode-reach theorem)
- Target theorem (axiom-free, via `TrustedBoundary`): a coarse-bandwidth
  Laplace field has a lower-bounded drift component toward an uncovered mode at
  distance `L`, uniformly in the presence of an added fine scale; connect to the
  metastability escape-time and to `LaplaceFissionInstability.lean`. This
  certifies the *directional-reach* lever (L6) exactly as T1/T4 certified the
  NCJ levers. Empirical confirmation remains M3/M4's job.

### M6 — Honest conclusion
| Outcome | Defensible claim |
|---|---|
| M3 passes and M4 transfers under Adam | A general mode-recovery improvement over the exact paper method that transfers through the tested generator — the win the program has been chasing |
| M3 passes, M4 fails under Adam | Particle-level mode-recovery improvement; optimizer-conditional (record why) |
| Only some `K`/`d`/init improve | Conditional design rule for hard multimodal regimes |
| M1 precondition fails | The regime was not hard enough; no claim; redesign |

Nothing here supports an ImageNet claim without a separate image-scale study.

## Execution order
1. Commit this roadmap.
2. M0: `mode_recovery.py` + metric invariant tests.
3. M1: freeze fresh registries; establish and commit the headroom precondition.
4. M2: two-scale field + invariant/compute tests; smoke only.
5. M3: run the pre-registered particle coverage gate; write results.
6. M4: cheap optimizer diagnostic; generator transfer only if it survives Adam
   and M3 passed.
7. M5: mode-reach theorem.
8. Update `ResearchStatus.md`, `DynamicsRoadmap.md`, and the ledger with the
   exact claim earned.

Milestone 1 is complete when M0–M1 are committed and the headroom precondition is
demonstrated: only then is there a proven problem worth attacking.
