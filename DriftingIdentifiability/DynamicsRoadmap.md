# The drifting-dynamics program: roadmap (2026-07-18)

*The focus document for the post-atlas phase.  End goal (agreed): the
defensible end-to-end claim — **"drifting provably converges to the data
distribution; here is the complete failure taxonomy; here are the certified
design rules that eliminate the failures."**  Evidence base: the completed
converse (`laplaceZeroDrift_identifies_euclidean`) + the three-pass Collapse
Atlas (`numerics/CollapseAtlas*.md`).  This file exists so sessions do not
wander: work through the phases in order, check off deliverables, resist new
tracks until the current phase's committed items are done.*

## Phase A — certify the atlas's dynamical statics (Lean, now)

The two cheap-but-sharp theorems the atlas earned.  Both are committed
deliverables; nothing else starts until they are green and audited.

* **A1 = T1, fission instability of point collapse (1-d).**
  New file `LaplaceFissionInstability.lean`.  Content:
  1. closed form for the mean shift of the symmetric pair
     `q_u = ½δ_{c+u} + ½δ_{c−u}` at its own atoms:
     `m_q(c±u) = ∓2u·k(2u)/(1+k(2u))` — an exact finite computation;
  2. at a root `m_p(c) = 0`, the certified two-sided derivative plus the
     certified `D′ = L/τ − 2Z` identity give the **fission-index formula**
     `m_p′(c) + 1 = (1/τ)·E_w|y−c|` (tilted mean absolute deviation);
  3. strict positivity: `m_p(c) = 0 ∧ p ≠ δ_c ⟹ m_p′(c) + 1 > 0` — the
     sharp form matching the atlas's P2B degeneracy boundary exactly;
  4. the dynamical reading: the pair-separation drift
     `g(u) = ½(V(c+u) − V(c−u))` satisfies `g(u)/u → m_p′(c)+1`, hence
     `g(u) > 0` for all small `u > 0` — **every point collapse at a
     non-atom of `p` is strictly fission-repelling.**
  Consumes: `hasDerivAt_laplaceMeanShiftRatio_of_root`, the 1-d
  displacement-derivative pair, two-atom integral computations.
* **A2 = D1, mass-blindness of finite-range kernels.**
  New file `FiniteRangeMassBlindness.lean`.  For a nonnegative radial
  kernel with `k = 0` beyond range `ρ` (and `k(0) > 0`), two-cluster
  measures with identical cluster shapes, different cluster masses, and
  separation `> 2ρ` have **identical drift fields everywhere** (the Lean
  `(0)⁻¹ • D = 0` convention covers the dead zone) — so the zero-drift
  converse **fails** for every finite-range kernel.  Sharp counterpoint to
  the Laplace theorem: the tail is load-bearing; RKHS-characteristicness is
  not the right dividing line for normalized drifts.
  (Numerically verified: `numerics/frontier_screen.py` [B].)

Acceptance for Phase A: both theorems `#print axioms`-clean, registered in
`AxiomAudit.ps1`, `Check.ps1` green, findings recorded in
`ResearchStatus.md`.

## Phase B — the quantitative layer (theory → Lean, next)

* **B1 = T2-lower, the residual floor.**  On imbalanced two-cluster states,
  `0 < ‖V‖_{supp q} ≤ C·e^{−L/τ}` — the lower bound formalizes "wrong-mass
  states are not equilibria" (atlas P3B/E3b evidence), the upper bound
  formalizes "but they are nearly so".  The lower bound is the first
  concrete instance of the quantitative-converse machinery (Direction B,
  the ε-perturbed maximum principle) and the entry point to it.
* **B2 = T2-upper, metastability bounds.**  Two-sided
  `e^{c₁L/τ} ≤ T_relax ≤ e^{c₂L/τ}` for the population flow (measured
  `c ≈ 1.7`).  Harder; attempt after B1 clarifies the constants.
* **B3 = T5, finite-N mask effects.**  The masked field's stationary shift:
  law-level `O(N^{-γ})` bound (measured γ ≈ 1.7) and the stable-wrong-
  equilibrium example at one particle per mode.  Start from the certified
  Obj-4 masked-vs-deleted analysis.

Phase B is *open-ended research*; timebox each item and record
falsifications in `ResearchStatus.md` rather than pushing indefinitely.

## Phase C — the performance demonstration (Python, after A; can interleave)

Wire the three atlas design rules into the estimator-level pipeline
(`numerics/driftlab.py` + the SNIS layer) and benchmark:

* **C1** bandwidth ladder (`τ' ≈ separation, equal weight`) vs the paper's
  grid; metric: relaxation/convergence speed + final ED/MMD on synthetic
  multimodal targets across dimensions.
* **C2** step-size rule `η* = min −2Reλ/|λ|²` (estimated from data) vs
  fixed steps.
* **C3** mask policy: eye-mask on/off × particles-per-mode; verify the
  stable-wrong-equilibrium hazard and its batch-size cure at estimator
  level (finite samples, not population).
* Acceptance: a `numerics/DesignRules.md` with measured
  improvement curves, seeds, and manifests — the "better performing model"
  evidence.

## Standing focus guards

1. Novelty discipline: Lee–Chun (2604.24196) owns informal priority on the
   converse for companion-elliptic kernels; our claims are *machine-checked
   proofs*, the *elementary max-principle route*, and *everything
   dynamical* (the atlas program — genuinely open territory).
2. No new research tracks (Matérn tower, ∞-dim, local-to-global, companion
   drifting) until Phase A is done and Phase B1 attempted; they stay in
   `PostConverseFrontier.md`.
3. Numerics-before-Lean for any new mathematical claim; evidence-calibrated
   language everywhere ("certified" = Lean only).
4. Every phase deliverable lands with: commit, audit entry (if Lean),
   `ResearchStatus.md` record, memory update.

## Status ledger

| Item | Status |
|---|---|
| A1 fission theorem | **DONE** (2026-07-18, axiom-clean, audited) |
| A2 mass-blindness theorem | **DONE** (2026-07-18, axiom-clean, audited) |
| B1 residual floor | queued |
| B2 metastability bounds | queued |
| B3 mask effects | queued |
| C1–C3 benchmarks | queued (can start any time after A) |
