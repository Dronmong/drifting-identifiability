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
| C1 bandwidth ladder | **VALIDATED, QUALIFIED** (coarse is essential; anneal vs fixed coarse is geometry-dependent) |
| C2 step size | **CORRECTED + VALIDATED** (full coupled generator is a ceiling, not an operating rule) |
| C3 mask policy | **VALIDATED, QUALIFIED** (small-N hazard transfers to paper bi-softmax; no stable wrong endpoint demonstrated) |

---

## Phase C audit and required validation pass (2026-07-19)

### Verdict

Phase C is a useful first estimator-level pilot, but the first pass is not
complete or reliable enough to mark C1--C3 `DONE`.

* **C1 bandwidth scheduling:** promising, but causal attribution needs repair.
* **C2 generator step rule:** mathematically incorrect as implemented.
* **C3 mask policy:** useful preliminary evidence, but incomplete.
* **Benchmark infrastructure:** a good start, but missing raw per-seed data and
  exact provenance.
* **Improvement over the paper estimator:** not yet supported.

The suite genuinely improves upon the population atlas: it draws a fresh
target minibatch each step, trains from generic particle clouds, records law
and mode diagnostics, uses paired deterministic seeds, and states that the
exact paper bi-softmax estimator and real features were not tested.  The
combined coarse-to-fine procedure also clearly performs better than a
permanently fine bandwidth on the synthetic targets.  The gaps below concern
attribution and strength of claim, not the absence of a useful signal.

### C1 audit: bandwidth scheduling

The intended question is whether coarse-to-fine bandwidth scheduling improves
mass transport without sacrificing final precision.  The implemented result
supports the weaker claim that the combined policy “start with large bandwidth
and large steps, then reduce both” performs well on the tested Gaussian
mixtures.  It does not isolate bandwidth scheduling itself.

#### Step-size confound

For fixed bandwidths, the step is proportional to that bandwidth.  For the
annealed arm it is recomputed as `eta_t = 0.1 * tau_t`.  Consequently:

* `single-L` starts with step `0.1 L`;
* `single-fine` uses approximately `0.1 tau_fine`;
* `ladder-eq` uses a step based on its smallest bandwidth;
* `paper-multi` also uses a step based on its smallest bandwidth;
* `anneal` begins with the much larger `0.1 L` step.

Thus `single-L` and `anneal` can move particles roughly an order of magnitude
farther per update.  The poor performance of averaging may be caused partly by
this discrepancy, not necessarily by averaging itself.

#### Baseline mismatch

The Phase C specification names the paper bandwidth set
`{0.02, 0.05, 0.2}`.  The code instead used the target-relative set
`{0.3 sigma, sigma, 0.3 L}`.  This can be a sensible synthetic baseline, but it
must not be called the paper's fixed set without implementing the paper's
feature normalization and actual values.

#### Oracle target geometry

The schedule receives the true mode separation `L` and width `sigma` from the
synthetic generator.  Those quantities must be estimated on realistic data.
The current rule is therefore tuning-free only conditional on oracle knowledge
of the target geometry.

#### Result wording and censoring

Annealing matched the grid baseline for K4/d1, but its final energy distance
was about three times worse for K4/d2 and K4/d5.  It reached a loose tolerance
rapidly, but “matches the oracle everywhere” is too strong.

Metrics were recorded only every 20 steps.  Reported values such as 10 steps
can be medians of observed crossing labels 0 and 20 rather than actual observed
crossing times.  Replacing censored runs by `9999` produced values such as
`5039`; this is not a valid time-to-event statistic.

#### Required C1 validation

1. Hold the same fixed `eta` across every bandwidth policy.
2. Separately compare the practical joint policy `eta_t = 0.1 tau_t`.
3. Equalize kernel evaluations and wall-clock budget: a three-bandwidth step
   costs roughly three single-bandwidth evaluations.
4. Implement the actual paper bandwidth set under explicit normalization.
5. Compare oracle `L,sigma` with values estimated only from target samples.
6. Use a broader held-out grid for `single-best`, evaluated under the same
   horizon and metric as the reported arms.
7. Record threshold crossing every step and use explicit censoring rather than
   sentinel-valued medians.

### C2 audit: step-size rule

This is the principal correctness problem.  The first implementation intended
to estimate the generator of the full empirical particle field but called
`drift_est(q[j:j+1], Y, tau)`.  Passing one particle makes the model law inside
`drift_est` a singleton, whose self mean shift is identically zero.  The
estimated derivative therefore resembles the derivative of the target mean
shift for a lone particle, not the `N*d` by `N*d` Jacobian of the coupled
particle system.

It omits:

* the effect of perturbing particle `j` on every other particle;
* the effect of `j` on the empirical `m_q`;
* cross-particle coupling;
* collective support modes.

The implementation also discarded eigenvalues with positive real part when
forming `eta*`.  An expanding generator direction cannot be stabilized by a
positive explicit-Euler step and must be reported, not ignored.  Therefore the
reported `2.67 tau` and `2.72 tau` are not estimates of the full generator
boundary.

The fixed-step sweep gives only limited evidence that steps of order `tau` are
reasonable.  In one dimension `1.0 tau` had the best final ED; in two
dimensions `0.1 tau` did, with small differences.  Two targets and four seeds
do not establish `0.5 tau` as universal.

#### Required C2 validation

Construct the complete frozen-batch field `F(q;Y) in R^(N*d)` and differentiate
with respect to flattened `q`.  For moderate `N`, central finite differences
are sufficient; larger systems can use Jacobian-vector products and Arnoldi.

1. Check whether every generator eigenvalue has negative real part.
2. Compute the Euler boundary from the full spectrum only in the stable case.
3. Test safety factors such as `0.1 eta*`, `0.25 eta*`, and `0.5 eta*`.
4. Compare against fixed `c tau` rules.
5. Include the computational cost of estimating the spectrum.
6. Include a large-bandwidth regime where the atlas observed Euler
   instability.

Until this is done, C2 is provisional.

### C3 audit: eye-mask policy

C3 contains a meaningful empirical signal.  At `N/K = 2`, masking performed
substantially worse and dropped a mode; at `N/K = 8` and `32`, masking
performed better in the simplified SNIS system.

The experiment does not establish that the masked endpoint is a stable wrong
equilibrium.  It records the endpoint after 800 steps but does not require a
small on-support drift residual, run a substantially longer horizon,
perturb-and-return, or inspect the local Jacobian.

The explanation that masking provides “variance reduction” is also not yet
demonstrated.  The model-side field uses the complete particle cloud, not a
stochastic minibatch of `q`.  Removing self-interaction primarily changes
leave-one-out bias and repulsion rather than ordinary Monte Carlo variance.

Most importantly, the Phase C specification requires both the simplified SNIS
estimator and `driftlab.compute_v_paper`.  Only SNIS was run.  The document
acknowledges this, but the status ledger nevertheless marked C3 complete.

#### Required C3 validation

1. Add the exact paper bi-softmax estimator.
2. Sweep `N/K` over at least `{1,2,4,8,16,32}`.
3. Independently sweep target minibatch size `B`.
4. Measure endpoint residual and local stability.
5. Continue suspicious endpoints for a much longer horizon.
6. Use several target families, including unequal mode weights.
7. Report the crossover as an interval; samples at only 2, 8, and 32 cannot
   locate it “around 8.”

### Reporting and reproducibility audit

The specification requested per-run trajectories, IQRs, censoring, and
configuration-bearing manifests.  The first pass saved only aggregated
medians.  Therefore the medians cannot be independently recomputed, censored
times are not handled correctly, and figures have no uncertainty bars.

The saved manifests reference commit `fba6d57`, which predates the addition of
`driftbench.py`.  The benchmark was run from an uncommitted working tree and
committed afterward, so the recorded commit does not reproduce the executed
source.

A corrected pass must save:

* commit, dirty status, source hash, command line, and full configuration;
* per-seed endpoint metrics and trajectories;
* wall time and kernel-evaluation count;
* paired uncertainty summaries and proper censored-time statistics.

Phase C must also be recorded in `ResearchStatus.md`, as required by the
standing deliverable rule.

### Revised Phase C status after audit

| Item | Audited status |
|---|---|
| C1 bandwidth ladder | **PROVISIONAL** — strong combined-policy signal; step-size-controlled validation required |
| C2 step size | **CORRECTION REQUIRED** — first `eta*` was not the coupled generator |
| C3 mask policy | **PROVISIONAL** — SNIS signal only; paper estimator and stability tests required |

The strongest currently defensible claim is:

> On the tested synthetic multimodal targets, a combined coarse-to-fine
> bandwidth-and-step schedule substantially outperforms a permanently fine
> kernel, and eye masking has a strong particle-count-dependent effect.

It is not yet defensible to claim tuning-free optimal rules or improvement
over the paper's exact estimator.  The validation pass above is required before
restoring `DONE` status.

### Validation pass completed (2026-07-19)

The required pass is implemented in `numerics/driftbench_v2.py`, specified in
`numerics/PhaseCValidation.md`, and reported in the rewritten
`numerics/DesignRules.md`.  Standard-profile raw runs, source snapshots,
manifests, per-seed CSVs, and trajectories live under
`numerics/bench_runs_v2/`; uncertainty figures live in
`numerics/bench_figs_v2/`.

The corrected conclusions are:

1. **C1 complete as an experiment, universal annealing claim rejected.**
   Coarse coupling reliably repairs missing-mode transport.  Annealing wins in
   the tested 2-D targets, ties fixed coarse in 1-D, and loses to fixed coarse
   in the tested 5-D target.  Unsupervised estimates of `L,sigma` match oracle
   scheduling on these clean mixtures.  Bandwidth-only and joint-step results
   are now reported separately at equal kernel cost.
2. **C2 implementation corrected.**  The full `N*d` generator includes all
   particle couplings and its spectral formula satisfies
   `rho(I + eta* J) = 1`.  The boundary is a ceiling and can be a poor training
   step; `0.1 eta*` performs well in the coarse regime, while no universal
   fraction of `tau` is established.
3. **C3 expanded to the paper estimator.**  The eye mask is decisively harmful
   at `N/K <= 2` and harmful on median at `N/K = 4` under both SNIS and the
   exact bi-softmax affinity.  Its median effect is near neutral at `N/K = 8`
   and configuration-dependent thereafter.  Across 480 endpoint rows and 134
   long continuations, zero states met the strengthened stable-wrong gate.

Phase C is therefore **complete as a synthetic validation phase**, with
qualified rather than universal design rules.  It does not establish
real-feature or neural-model improvement, and it does not replace the open
Phase-B convergence theory.

### Low-dimensional matched follow-up (2026-07-19)

The matched exact-Algorithm-2 program in
`numerics/LowDimPerformanceRoadmap.md` failed its original aggregate gate.  A
post-audit fresh factorial follow-up (`LowDimAttributionProtocol.md`) corrected
the inference, provenance, metric, and factor-confounding gaps.  Its global
gate also failed, so D4 learned-generator transfer remains blocked.

The follow-up nevertheless isolates one conditional design mechanism:
geometry-matched bandwidth and scaled steps do not beat the tuned base on the
fresh validation suite, while conditionally disabling the eye mask cuts ED²
by roughly 32--52% on fresh ring/circle/moon targets.  The provisional
cluster-count trigger can harm an overclustered Gaussian mixture and is not a
finished algorithm.  See `numerics/LowDimAttributionResults.md`.  Any next
performance attempt must use a new split and should test an affinity-leverage
or self-mass trigger; Phase B1 may proceed independently but cannot yet be
presented as explaining an aggregate performance win.

### NCJ identifiability-driven program (2026-07-20)

`numerics/IdentifiabilityDrivenImprovementPlan.md` replaced the rejected
mask-trigger policy with a mechanism-separated candidate — normalized (drop the
`P*Q` gain), cross-fitted (independent reference batch, no eye mask), jittered
(symmetric Gaussian) drifting — backed by four trust-audited theorems
(`NCJIdentifiability.lean`, T1–T4).

**Outcome: the pre-registered particle gate PASSED; the pre-registered
learned-generator gate FAILED.**  The earned claim (plan §9) is an
empirical-particle estimator improvement over the exact paper implementation,
under matched architecture and compute, that does **not** transfer through the
tested learned generator.

* E4 particle gate PASS: target-balanced geo-mean ED² ratio **0.100**,
  hierarchical CI `[0.050, 0.195]`, 93.75% winning cells, every family < 1, all
  eight criteria met (`20260720-011000-NCJ-test-standard/e4_gate.json`).
* E5 generator gate FAIL: ratio **1.072** vs paper, CI `[0.946, 1.208]`, 28% of
  cells winning; the constant-gain field is an excellent free-particle mover but
  not a better MLP regression target under the paper loop
  (`20260720-024712-NCJ-generator-standard/e5_gate.json`).
* Mechanism attribution (validation): dropping `P*Q` is dominant (0.31×),
  cross-fitting compounds it (→ 0.088×); **jitter never helped** and was frozen
  out at `sigma=0`.  Consequently **T5 (jitter/fission) is intentionally not
  formalized** — the data give it no support.

See `numerics/IdentifiabilityImprovementResults.md`.  A learned-generator
improvement remains unearned; the next attempt must redesign the field→generator
coupling on a fresh split rather than reuse this particle policy.

### Next target: mode-recovery program (2026-07-20)

The cross-program ledger
(`docs/04-empirical-experimentation/PaperImprovementAttempts.md`) shows aggregate
error on easy, well-tuned, low-dimensional targets has no headroom, but
**missing-mode / mode-coverage recovery** is the one axis with real, repeated,
non-conditional wins (atlas two-bandwidth 31–38×; coarse bandwidth; NCJ
missing-mode KM 60 vs 114).  The active roadmap
`numerics/ModeRecoveryRoadmap.md` reframes the objective to a **coverage gate**
in a regime where one fixed bandwidth cannot both reach and resolve, with a
fixed multi-scale field, a headroom precondition (M1), an early
optimizer diagnostic (the NCJ lesson), and a directional-reach transfer
hypothesis (why it may survive Adam where NCJ did not) plus a mode-reach
theorem target (M5).
