# Phase C: the certified design rules, benchmarked

> Historical procedure (superseded 2026-07-19). This file specifies the
> original Phase C pass. Its conclusions were audited and rerun with corrected
> controls, the full coupled generator, the exact paper bi-softmax estimator,
> per-seed trajectories, and reproducibility manifests. Use
> `PhaseCValidation.md` for the current protocol and `DesignRules.md` for the
> validated, qualified conclusions. The original files are retained so the
> correction is auditable.

*Procedure spec (2026-07-19).  Implements Phase C of
`DriftingIdentifiability/DynamicsRoadmap.md` — the "better performing model"
leg of the tripod.  The Collapse Atlas established the three design rules at
the **population** level (exact field).  Phase C tests whether they survive
at the **finite-sample / estimator** level, on genuinely multimodal targets,
and measures end-to-end generative quality.  Companion code
`numerics/driftbench.py`; runs in `numerics/bench_runs/<id>/`; findings in
`numerics/DesignRules.md`.*

---

## 1. What we are measuring, and against what

The atlas measured **relaxation speed** of an already-correct configuration
under the *exact* population field.  Real training instead:

* estimates the drift from **finite minibatches** (self-normalized, noisy);
* starts from a **generic init**, not a near-solution;
* must **cover all modes** of a multimodal target, not just relax one.

Phase C asks, for each rule: does it improve convergence speed and final
generative quality of the *actual finite-sample training loop*, versus the
paper's defaults, across dimensions and target geometries?

## 2. The training loop (faithful finite-sample drifting)

Target `p` = data distribution (sampled).  Model `q_t = {x_1,…,x_N}` a
particle cloud, equal weights.  Per step:

1. draw a data minibatch `Y ⊂ p`, `|Y| = B`;
2. for each particle `x_j`, estimate the mean-shift drift
   `V̂(x_j) = m̂_p(x_j) − m̂_q(x_j)` where
   `m̂_r(x) = Σ_i k(x,y_i)(y_i − x) / Σ_i k(x,y_i)` (SNIS mean shift),
   `m̂_p` over `Y`, `m̂_q` over the other particles (optional eye-mask);
3. `x_j ← x_j + η · V̂(x_j)`.

Kernel `k(x,y) = exp(−‖x−y‖₂/τ)` (the paper's ℓ² Laplace kernel).  This is
the honest finite-sample analogue of the population dynamics the atlas and
the converse both analyze; the paper's bi-softmax affinity is one specific
estimator of the same field (used in C3's mask arm as a cross-check).

## 3. Targets and metrics

**Targets** (`p`): mixtures of `K` isotropic Gaussians in `d` dims, modes on
a scaled simplex/grid, separation `L`, width `σ = 0.15·L`.  Families:
`K∈{2,4,8}`, `d∈{1,2,5}`, `L/τ` spanning the metastable regime.

**Metrics** (all exact/sample-based, no learned critic):

* **energy distance** `ED(q_t, p)` (sample estimate against a fresh
  reference sample of `p`) — a genuine metric on laws;
* **mode coverage**: assign each particle to its nearest target mode; count
  modes holding `≥ 0.5/K` of the particle mass (a covered mode) → fraction
  in `[0,1]`;
* **mode-mass error**: `Σ_k |q̂(mode k) − 1/K|` (detects the atlas's
  metastable mass-imbalance failure directly);
* **steps-to-target**: first step with `ED < ED_tol` (censored if never).

Each cell: multiple seeds; report medians + IQR; censoring recorded.

## 4. Experiments

### C1 — the bandwidth ladder (flagship)

Arms, all else equal:

* **single-best**: single bandwidth `τ`, grid-searched to minimize final
  `ED` per target (the paper's per-target tuning, best case for the baseline);
* **paper-multi**: the paper's fixed 3-bandwidth set `{0.02, 0.05, 0.2}`
  (normalized-distance scale), averaged;
* **ladder**: the atlas rule — `{τ_fine, τ' ≈ L}` with **equal weight**,
  `τ_fine` small for sharp final placement, `τ'` at the mode-separation
  scale to erase the `e^{L/τ}` barrier.

**Hypothesis C1.**  The ladder reaches full mode coverage and low mode-mass
error in far fewer steps than single-best (which suffers metastable
imbalance at small `τ`) and beats the fixed paper-multi set (which is not
matched to `L`).  Predicted from atlas P2C/P3E (31–38× population speedup)
— C1 tests survival under estimator noise and generic init.

### C2 — the step-size rule

Arms: fixed `η ∈ {0.1τ, 0.5τ, τ}` (grid, baseline) vs the generator rule
`η* = min −2Reλ/|λ|²` estimated online from a finite-difference of the
current empirical field near the particle cloud (or the atlas closed-form
`η* ≈ c·τ` with `c` read from a cheap local probe).

**Hypothesis C2.**  The generator-derived `η*` matches the best grid `η`
without a sweep and avoids the Euler-overshoot divergence the atlas found
at large `τ` (P2E) — a tuning-free learning rate.

### C3 — the mask policy

Arms: eye-mask on/off × particles-per-mode `N/K ∈ {2, 8, 32}`, using both
the SNIS mean-shift estimator and the paper's bi-softmax affinity
(`driftlab.compute_v_paper`) as the drift.

**Hypothesis C3.**  At small `N/K` the eye-mask induces the stable-wrong
equilibrium the atlas found (P3A): final `ED`/mode-mass error is *worse*
with the mask and improves as `N/K` grows (the certified `O(1/N)` shift).
Design rule: drop the mask, or ensure `N/K` large.

## 5. Deliverables

1. `driftbench.py` — the training harness + all three experiments,
   deterministic seeds, run manifests (commit, versions, config).
2. `bench_runs/<id>/` per run: `manifest.json`, metric trajectories
   (`.npz`), endpoint table (`.csv`), `summary.md`.
3. `DesignRules.md` — curated improvement curves + tables + figures
   (`numerics/bench_figs/`): the "certified design rules that measurably
   improve the model" evidence.
4. A short honest-scope note: these are synthetic multimodal targets with
   exact metrics; real-data/encoder-feature validation is out of scope
   (flagged, not claimed).

## 6. Reproducibility

```
uv run --with numpy --with scipy --with matplotlib python numerics/driftbench.py [C1|C2|C3|all]
```

Master seed 20260719; per-experiment reseed.  Evidence-calibrated wording
throughout ("certified" = Lean only; here everything is measured evidence).
