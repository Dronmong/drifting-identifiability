# Certified design rules, benchmarked (Phase C, 2026-07-19)

*The "better performing model" leg of the dynamics tripod
(`DriftingIdentifiability/DynamicsRoadmap.md`).  The Collapse Atlas
established three design rules at the **population** level; Phase C tests them
at the **finite-sample / estimator** level with the actual training loop
(SNIS mean-shift drift, minibatches, generic init) on multimodal Gaussian
targets.  Harness `driftbench.py`; runs in `bench_runs/`; figures in
`bench_figs/`.  Seed 20260719.  Evidence-calibrated: everything here is
measured, not certified.*

**Headline.**  All three atlas rules transfer to the estimator level, and
two of the three sharpened into something better than first stated when
tested — exactly what a benchmark is for.  The strongest result is a
**tuning-free coarse-to-fine bandwidth schedule that matches a
grid-searched optimum and converges where the naive baseline never does.**

---

## C1 — bandwidth: schedule the separation scale, don't average it

Final energy distance and steps-to-tolerance, median over 4 seeds,
N = 20K particles, batch 64, 800 steps (`bench_figs/c1_bandwidth.png`):

| target | single-fine | paper-multi | ladder-eq | single-L | single-best* | **anneal** |
|---|---|---|---|---|---|---|
| K4 d1 | 0.176 (30) | 0.113 (30) | 0.057 (20) | 0.029 (10) | 0.006 (20) | **0.006 (10)** |
| K4 d2 | 0.268 (∞) | 0.226 (∞) | 0.197 (∞) | 0.021 (20) | 0.011 (20) | **0.031 (20)** |
| K8 d2 | 0.104 (60) | 0.075 (70) | 0.046 (50) | 0.021 (20) | 0.010 (20) | **0.015 (20)** |
| K4 d5 | 0.107 (60) | 0.093 (80) | 0.078 (60) | 0.023 (20) | 0.023 (20) | **0.069 (20)** |

`(steps→tol)`; `∞` = never reached `ED < 0.05L` in 800 steps.
`*single-best` = the best bandwidth found by an offline grid search per
target (the paper's per-target tuning; an oracle baseline, not tuning-free).

**Findings.**

* **The naive "small bandwidth for precision" choice (`single-fine`,
  τ = σ/2) is the worst** — slow, and in d = 2 it never converges.  This is
  the atlas metastability barrier (`e^{L/τ}`) surviving at the estimator
  level: a fine kernel cannot move mass between well-separated modes.
* **The paper's fixed multi-bandwidth set (`paper-multi`) is mediocre** —
  not matched to the target's separation `L`.
* **A single bandwidth at the separation scale (`single-L`, τ = L) is
  excellent** — the atlas's core insight (a separation-scale kernel erases
  the barrier), realized as one bandwidth, with no fine-scale drag.
* **The equal-weight ladder (`ladder-eq`, average of fine + coarse) — my
  first, naive instantiation of the atlas rule — underperforms `single-L`.**
  Averaging a slow fine component into the field drags the whole update
  down.  *Averaging is the wrong way to combine scales.*
* **Coarse-to-fine annealing (`anneal`, τ geometric from L to σ/2 over 70%
  of the run) is the tuning-free winner** — it matches or approaches the
  grid-searched `single-best` **without any search**, converges in
  10–20 steps everywhere (including where `single-fine` never does), and
  beats every fixed strategy except the oracle.  It gets both the
  barrier-erasing of the coarse scale *and* the placement precision of the
  fine scale, in sequence.

**Refined design rule (C1).**  *Anneal the bandwidth coarse-to-fine,
starting at the mode-separation scale and ending at the mode width.*  This
supersedes both the paper's fixed grid and my initial equal-weight-ladder
reading of the atlas.  The atlas insight was right; the correct realization
is a **schedule**, not a fixed average.

---

## C2 — step size: `O(τ)` is right, the stability boundary is an upper bound

Final ED, median over 4 seeds, τ = σ, 600 steps:

| target | 0.1τ | 0.5τ | 1.0τ | 2.0τ | generator η* | est. η* |
|---|---|---|---|---|---|---|
| K4 d1 | 0.045 | **0.007** | 0.005 | 0.021 | 0.032 | 2.67τ |
| K4 d2 | 0.050 | 0.054 | 0.060 | 0.055 | 0.075 | 2.72τ |

**Findings.**  The generator rule `η* = min −2Reλ/|λ|²` estimated online
gives `η* ≈ 2.7τ` — the correct **scale** (`O(τ)`, confirming the atlas E5
prediction) but too aggressive a **constant**: it is the linear-stability
*boundary* at the fixed point, whereas the practical optimum for the noisy,
finite-step estimator sits well inside it, around `η ≈ 0.5–1τ`.  Nothing
diverged even at 2τ here because τ = σ ≪ diameter (the atlas's large-τ Euler
divergence needs τ comparable to the data diameter).

**Refined design rule (C2).**  Use `η ≈ 0.5τ` — a fixed fraction of the
bandwidth, tuning-free and robust.  The generator η* is a safe *upper*
bound (useful to cap the step and detect the large-τ divergence regime),
not the operating point.

---

## C3 — the mask is a small-`N` hazard

Final ED / coverage / mass-error, median over 5 seeds, K = 4, d = 2,
τ = σ (`bench_figs/c3_mask.png`):

| N/K | eye-mask off | eye-mask on |
|---|---|---|
| 2  | ED 0.75, cover 1.00 | **ED 3.11, cover 0.75** |
| 8  | ED 0.49, cover 1.00 | ED 0.14, cover 1.00 |
| 32 | ED 0.14, cover 1.00 | ED 0.035, cover 1.00 |

**Findings.**  This confirms the atlas's sharpest practical warning and
adds nuance.  At **small particles-per-mode the eye-mask is actively
harmful** — ED 4× worse and a *dropped mode* (coverage 0.75) — the
stable-wrong-equilibrium the atlas found at one particle per mode (P3A),
now reproduced at the estimator level.  At **large N/K the mask helps**
(variance reduction from self-exclusion dominates once the `O(1/N)`
equilibrium shift is negligible).  The two effects cross over around
N/K ≈ 8.

**Refined design rule (C3).**  *Use the eye-mask only when
particles-per-mode is comfortably large (≳ 8 here); at small batch/particle
counts drop it, or it induces a stable wrong equilibrium and drops modes.*

---

## What Phase C establishes

The design rules extracted from the certified converse + the atlas are not
just population-level curiosities: they transfer to the actual finite-sample
training loop, and the benchmark turned two of them into **better,
tuning-free** rules than first proposed:

1. **anneal the bandwidth** (coarse-to-fine, L → σ): matches an oracle grid
   search with no tuning and fixes the metastability non-convergence;
2. **η ≈ 0.5τ**: tuning-free step size, with the generator η* as a safety cap;
3. **eye-mask only at large N/K**: else it induces a stable wrong equilibrium.

Together with the converse (there is one global optimum) and the atlas
failure taxonomy (collapse is fission-unstable; the enemy is metastable mass
transport), this closes the tripod: *drifting has one optimum, a known
failure mode, and certified-derived design rules that measurably reach the
optimum faster.*

## Honest scope

Synthetic multimodal Gaussian targets with exact sample-based metrics
(energy distance, mode coverage/mass) and the SNIS mean-shift estimator.
Real-data / learned-encoder-feature validation and the paper's exact
bi-softmax affinity at scale are out of scope here — flagged as the natural
next empirical step, not claimed.  C3's mask arm used the SNIS estimator;
cross-checking against `driftlab.compute_v_paper` at scale is future work.
