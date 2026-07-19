# Collapse Atlas — findings summary (2026-07-18)

*Curated conclusions; raw experiment logs appended below by
`collapse_atlas.py` (master seed 20260718).  Spec: `CollapseAtlas.md`.
The E3(b) residual-floor evidence (local least-squares; not a certificate) is in `atlas_e3b_check.py`.*


> **Audit note (2026-07-18).**  `CollapseAtlasAudit.md` audited this pass.
> Known pass-1 caveats: E4 ran a reduced matrix (P1 only, 60 inits/cell,
> N in {8,32}, absolute spreads, squared-energy-distance threshold, no
> residual criterion); E5 omitted the promised self-mask comparison; E6
> tested the Euler map, not the flow (increases may be discretization
> overshoot -- re-tested at the flow level in pass 2); the E3 exponent fit
> ignores censoring and conflates sigma/tau with L/tau; "certified"
> language has been downgraded to evidence.  Pass-2 protocol:
> `CollapseAtlas.md` (Pass 2); runs live in `numerics/atlas_runs/`.

## Headline findings

1. **(E1) The found spurious-equilibrium landscape matches the mean-shift mode
   structure** (multistart census; not exhaustive).  Zeros of `m_p` = sinks (one per resolved cluster) plus
   saddles/sources between them; sinks merge into one as `τ` passes the
   separation scale.  Lone-particle collapse candidates exist at every
   bandwidth.

2. **(E2) Collapse is always fission-unstable — the dichotomy conjecture
   survives adversarial attack.**  The pair-splitting linearization
   `u ← u + η·(I + Dm_p(c))u` is verified numerically (relative error at
   the perturbation scale).  The splitting index
   `σ(c) = max Re eig(I + Dm_p(c))` was positive at **every found** sink of every
   family in d = 1, 2 (census minimum `+0.018`), and 24 adversarial
   Nelder–Mead searches over 4-atom 2-d configurations pinned to exact
   zeros could push it only to `+0.005`, never negative.
   *Conjecture (A2-dichotomy):* `σ ≥ 0` always, `> 0` unless degenerate;
   collapse can be made arbitrarily *slow* to escape but never trapping.
   **1-d theorem target (near-free in Lean):** `σ = 1 + m_p′(c) =
   (1/τ)·Cov_w(X, sgn X) ≥ 0` follows from the already-certified
   `laplaceMeanShiftRatioDeriv_add_one_nonneg`; strictness under a
   non-degeneracy hypothesis.

3. **(E3 + atlas_e3b_check) Wrong-mass states appear NOT to be equilibria — but they
   are exponentially metastable.**  Deterministic halving time of a
   two-cluster mass imbalance grows like `exp(c·L/τ)` with fitted
   `c ≈ 1.5` over `L/τ ∈ [2, 5]`, and stalls beyond (`> 12000·η` at
   `L/τ = 5.9`).  At `L/τ ≈ 7`, least-squares gives **numerical evidence for a strictly
   positive residual floor** (`‖V‖ ≈ 3.7e-6 > 0`, gradient-converged): the
   stalled state is a slow *traveling* state, not a fixed point.  So for
   finite atomic supports the only exact equilibria found are `p` itself
   and fission-unstable point collapses — **the practical failure mode of
   drifting is metastable mass imbalance, not collapse and not spurious
   equilibria.**  This gives the paper's multi-bandwidth heuristic a
   precise role: a large-`τ` component removes the `exp(c·L/τ)` barrier.

4. **(E4) Generic robustness.**  From 720 random initializations across
   `(τ, N, spread)` at `L/τ ≤ 5`: zero collapses, zero metastable
   endpoints, ≥ 95% full convergence (the rest still in transit).  Collapse
   basins are not reachable from generic inits — consistent with fission
   instability.

5. **(E5) The truth is contractively stable, with a bandwidth-dependent
   step-size ceiling.**  At `q = p` the update map has spectral radius
   `< 1` for `η ≤ 0.5τ` at all bandwidths tested; at `τ` comparable to the
   diameter the map destabilizes between `η = 0.5τ` and `η = τ`
   (`ρ = 1.48` at `η = τ`), while small `τ` tolerated `η = 2τ`.
   Quantifying `η*(τ)` is a clean follow-up with practical value.

6. **(E6) No global Lyapunov functional among the natural candidates.**
   Over 948 tracked intervals: energy distance and Laplace-MMD² each
   increase on 4.7% of intervals, smoothed-mass gap 7.5%, companion defect
   7.0%, potential-ratio oscillation 11.0%.  Consistent with the
   non-conservative-field residual in the finite-particle literature; the
   A3 question shifts toward either a cleverer functional or an
   impossibility statement.

## Extracted theorem targets

* **T1 (Lean, cheap):** 1-d fission instability of point collapse from the
  certified tilted-mean derivative bound.
* **T2 (theory):** the metastability law — lower/upper bounds
  `exp(c₁ L/τ) ≤ T_relax ≤ exp(c₂ L/τ)` for two-cluster targets; the
  residual floor `0 < ‖V‖ ≲ e^{−L/τ}` on imbalanced states (the
  lower-bound side is the quantitative-converse machinery of Direction B
  applied locally).
* **T3 (theory, open):** no-spurious-equilibria for finite atomic `q ≠ p` — is
  the observed positive residual floor a theorem?  (Connects to the
  support-restricted converse A1: for finite supports the answer would be
  "only collapses and `p` are stationary".)
* **T4 (numerics → theory):** the step-size stability boundary `η*(τ)`.


# Collapse Atlas run (E1), seed 20260718

## E1 fixed-point census

```
E1: fixed-point census of m_p (collapse candidates)
  d=1 P1 tau= 0.5: zeros=3 sinks=2 kinds=['sink', 'sink', 'source']
  d=1 P1 tau= 1.0: zeros=3 sinks=2 kinds=['sink', 'sink', 'source']
  d=1 P1 tau= 2.5: zeros=1 sinks=1 kinds=['sink']
  d=1 P1 tau= 6.0: zeros=1 sinks=1 kinds=['sink']
  d=1 P2 tau= 0.5: zeros=3 sinks=2 kinds=['sink', 'sink', 'source']
  d=1 P2 tau= 1.0: zeros=3 sinks=2 kinds=['sink', 'sink', 'source']
  d=1 P2 tau= 2.5: zeros=1 sinks=1 kinds=['sink']
  d=1 P2 tau= 6.0: zeros=1 sinks=1 kinds=['sink']
  d=1 P3 tau= 0.5: zeros=5 sinks=3 kinds=['source', 'sink', 'source', 'sink', 'sink']
  d=1 P3 tau= 1.0: zeros=1 sinks=1 kinds=['sink']
  d=1 P3 tau= 2.5: zeros=1 sinks=1 kinds=['sink']
  d=1 P3 tau= 6.0: zeros=1 sinks=1 kinds=['sink']
  d=1 P4 tau= 0.5: zeros=1 sinks=1 kinds=['sink']
  d=1 P4 tau= 1.0: zeros=1 sinks=1 kinds=['sink']
  d=1 P4 tau= 2.5: zeros=1 sinks=1 kinds=['sink']
  d=1 P4 tau= 6.0: zeros=1 sinks=1 kinds=['sink']
  d=2 P1 tau= 0.5: zeros=3 sinks=2 kinds=['sink', 'saddle', 'sink']
  d=2 P1 tau= 1.0: zeros=3 sinks=2 kinds=['sink', 'saddle', 'sink']
  d=2 P1 tau= 2.5: zeros=1 sinks=1 kinds=['sink']
  d=2 P1 tau= 6.0: zeros=1 sinks=1 kinds=['sink']
  d=2 P2 tau= 0.5: zeros=3 sinks=2 kinds=['saddle', 'sink', 'sink']
  d=2 P2 tau= 1.0: zeros=3 sinks=2 kinds=['sink', 'sink', 'saddle']
  d=2 P2 tau= 2.5: zeros=1 sinks=1 kinds=['sink']
  d=2 P2 tau= 6.0: zeros=1 sinks=1 kinds=['sink']
  d=2 P3 tau= 0.5: zeros=5 sinks=3 kinds=['sink', 'saddle', 'sink', 'sink', 'saddle']
  d=2 P3 tau= 1.0: zeros=1 sinks=1 kinds=['sink']
  d=2 P3 tau= 2.5: zeros=1 sinks=1 kinds=['sink']
  d=2 P3 tau= 6.0: zeros=1 sinks=1 kinds=['sink']
  d=2 P4 tau= 0.5: zeros=5 sinks=3 kinds=['sink', 'sink', 'sink', 'saddle', 'saddle']
  d=2 P4 tau= 1.0: zeros=3 sinks=2 kinds=['sink', 'sink', 'saddle']
  d=2 P4 tau= 2.5: zeros=1 sinks=1 kinds=['sink']
  d=2 P4 tau= 6.0: zeros=1 sinks=1 kinds=['sink']
```

# Collapse Atlas run (E2), seed 20260718

## E2 splitting index

```
E2: splitting index sigma = max Re eig(I + Dm_p) at sinks
  d=1 P1 tau= 0.5 sink=[5.174] sigma=+0.1675 linerr=1.00e-06
  d=1 P1 tau= 0.5 sink=[0.107] sigma=+0.3691 linerr=1.00e-06
  d=1 P1 tau= 1.0 sink=[5.136] sigma=+0.1135 linerr=1.00e-06
  d=1 P1 tau= 1.0 sink=[0.208] sigma=+0.3000 linerr=1.00e-06
  d=1 P1 tau= 2.5 sink=[3.107] sigma=+0.9660 linerr=1.00e-06
  d=1 P1 tau= 6.0 sink=[2.687] sigma=+0.4156 linerr=1.00e-06
  d=1 P2 tau= 0.5 sink=[0.108] sigma=+0.3710 linerr=1.00e-06
  d=1 P2 tau= 0.5 sink=[5.175] sigma=+0.1673 linerr=1.00e-06
  d=1 P2 tau= 1.0 sink=[5.16] sigma=+0.0955 linerr=1.00e-06
  d=1 P2 tau= 1.0 sink=[0.304] sigma=+0.3998 linerr=1.00e-06
  d=1 P2 tau= 2.5 sink=[4.817] sigma=+0.2664 linerr=1.00e-06
  d=1 P2 tau= 6.0 sink=[4.137] sigma=+0.2746 linerr=1.00e-06
  d=1 P3 tau= 0.5 sink=[5.055] sigma=+0.0787 linerr=1.00e-06
  d=1 P3 tau= 0.5 sink=[0.242] sigma=+0.4055 linerr=1.00e-06
  d=1 P3 tau= 0.5 sink=[2.585] sigma=+0.3275 linerr=1.00e-06
  d=1 P3 tau= 1.0 sink=[2.569] sigma=+0.4992 linerr=1.00e-06
  d=1 P3 tau= 2.5 sink=[2.594] sigma=+0.4566 linerr=1.00e-06
  d=1 P3 tau= 6.0 sink=[2.612] sigma=+0.2425 linerr=1.00e-06
  d=1 P4 tau= 0.5 sink=[0.11] sigma=+0.0809 linerr=1.00e-06
  d=1 P4 tau= 1.0 sink=[0.175] sigma=+0.1634 linerr=1.00e-06
  d=1 P4 tau= 2.5 sink=[0.392] sigma=+0.2105 linerr=1.00e-06
  d=1 P4 tau= 6.0 sink=[0.539] sigma=+0.1213 linerr=1.00e-06
  d=2 P1 tau= 0.5 sink=[0.145 0.174] sigma=+0.3654 linerr=1.00e-06
  d=2 P1 tau= 0.5 sink=[4.857 0.141] sigma=+0.3834 linerr=1.00e-06
  d=2 P1 tau= 1.0 sink=[4.814 0.136] sigma=+0.2553 linerr=1.00e-06
  d=2 P1 tau= 1.0 sink=[0.251 0.159] sigma=+0.3101 linerr=1.00e-06
  d=2 P1 tau= 2.5 sink=[2.529 0.138] sigma=+0.9291 linerr=1.00e-06
  d=2 P1 tau= 6.0 sink=[2.553 0.145] sigma=+0.3901 linerr=1.00e-06
  d=2 P2 tau= 0.5 sink=[0.146 0.174] sigma=+0.3687 linerr=1.00e-06
  d=2 P2 tau= 0.5 sink=[4.858 0.141] sigma=+0.3834 linerr=1.00e-06
  d=2 P2 tau= 1.0 sink=[4.858 0.139] sigma=+0.2394 linerr=1.00e-06
  d=2 P2 tau= 1.0 sink=[0.381 0.147] sigma=+0.4312 linerr=1.00e-06
  d=2 P2 tau= 2.5 sink=[4.487 0.131] sigma=+0.2792 linerr=1.00e-06
  d=2 P2 tau= 6.0 sink=[3.886 0.138] sigma=+0.2630 linerr=1.00e-06
  d=2 P3 tau= 0.5 sink=[4.928 0.053] sigma=+0.5312 linerr=1.00e-06
  d=2 P3 tau= 0.5 sink=[2.445 0.122] sigma=+0.2915 linerr=1.00e-06
  d=2 P3 tau= 0.5 sink=[0.24 0.1 ] sigma=+0.4852 linerr=1.00e-06
  d=2 P3 tau= 1.0 sink=[2.405 0.115] sigma=+0.4744 linerr=1.00e-06
  d=2 P3 tau= 2.5 sink=[2.46  0.104] sigma=+0.4435 linerr=1.00e-06
  d=2 P3 tau= 6.0 sink=[2.511 0.101] sigma=+0.2368 linerr=1.00e-06
  d=2 P4 tau= 0.5 sink=[-0.363  1.642] sigma=+0.5511 linerr=1.00e-06
  d=2 P4 tau= 0.5 sink=[-1.982 -0.257] sigma=+0.3982 linerr=1.00e-06
  d=2 P4 tau= 0.5 sink=[3.093 0.119] sigma=+0.0180 linerr=1.00e-06
  d=2 P4 tau= 1.0 sink=[2.87  0.192] sigma=+0.4430 linerr=1.00e-06
  d=2 P4 tau= 1.0 sink=[-0.571  1.355] sigma=+0.5985 linerr=1.00e-06
  d=2 P4 tau= 2.5 sink=[-0.15   1.009] sigma=+0.4025 linerr=1.00e-06
  d=2 P4 tau= 6.0 sink=[0.163 0.827] sigma=+0.2359 linerr=1.00e-06
  minimum sigma over census: +0.01798
  adversarial search (2-d, 4 atoms + c, 24 restarts):
  adversarial minimum sigma at exact zeros: +0.00500
```

# Collapse Atlas run (E3), seed 20260718

## E3 metastability and spurious equilibria

```
E3: mass-imbalance relaxation (deterministic, then minibatch)
  (a) deterministic dynamics, tau sweep:
    tau=1.0: alpha 0.2 -> 0.325 (target 0.5), final on-support resid=2.96e-02
    tau=1.7: alpha 0.2 -> 0.500 (target 0.5), final on-support resid=2.56e-04
    tau=2.5: alpha 0.2 -> 0.500 (target 0.5), final on-support resid=2.48e-04
  (b) Newton-polish the stuck state -> exact spurious equilibrium?
    Newton success=False resid=2.54e-04 alpha=0.500 (p has 0.5) -> no spurious equilibrium found
    spectral radius of update map at spurious eq: 1.002168 (unstable)
  (c) minibatch dynamics: steps to halve the imbalance
    L/tau=2.94 B=16: halving times [88, 51, 45]
    L/tau=2.94 B=64: halving times [98, 73, 78]
    L/tau=2.00 B=16: halving times [26, 27, 13]
    L/tau=2.00 B=64: halving times [31, 24, 22]
    L/tau=1.52 B=16: halving times [20, 20, 10]
    L/tau=1.52 B=64: halving times [14, 16, 14]
```

# Collapse Atlas run (E3), seed 20260718

## E3 metastability and spurious equilibria

```
E3: mass-imbalance relaxation (deterministic transport law)
  (a) deterministic halving flow-time T vs L/tau (hypothesis: log T linear in L/tau):
    L/tau= 2.00: flow-time to halve =      7.50
    L/tau= 2.50: flow-time to halve =     10.40
    L/tau= 2.94: flow-time to halve =     15.64
    L/tau= 3.57: flow-time to halve =     34.02
    L/tau= 4.17: flow-time to halve =     85.92
    L/tau= 5.00: flow-time to halve =    809.20
    L/tau= 5.88: flow-time to halve =    >12000*eta (stalled)
    fitted slope d(log T)/d(L/tau) = 1.519 (pure exponential law would be ~1)
  (b) spurious-equilibrium hunt at large separation (L/tau ~ 7):
    after 30000 steps: alpha=0.200 resid=1.10e-05
    Newton: success=False resid=1.05e-05 alpha=0.200 -> no exact spurious equilibrium
```

# Collapse Atlas run (E5), seed 20260718

## E5 stability at truth

```
E5: spectral radius of the update map at q = p, vs eta/tau
  tau=1.0 eta=0.05*tau: rho(J)=0.99615 (stable)
  tau=1.0 eta=0.2*tau: rho(J)=0.98459 (stable)
  tau=1.0 eta=0.5*tau: rho(J)=0.96147 (stable)
  tau=1.0 eta=1.0*tau: rho(J)=0.92294 (stable)
  tau=1.0 eta=2.0*tau: rho(J)=0.98854 (stable)
  tau=2.5 eta=0.05*tau: rho(J)=0.99681 (stable)
  tau=2.5 eta=0.2*tau: rho(J)=0.98725 (stable)
  tau=2.5 eta=0.5*tau: rho(J)=0.96812 (stable)
  tau=2.5 eta=1.0*tau: rho(J)=1.47582 (UNSTABLE)
  tau=2.5 eta=2.0*tau: rho(J)=3.95164 (UNSTABLE)
```

# Collapse Atlas run (E6), seed 20260718

## E6 Lyapunov screen

```
E6: monotonicity violation rates of Lyapunov candidates
  ED       : increased on 45/948 intervals (4.7%)
  MMD2     : increased on 45/948 intervals (4.7%)
  Zgap     : increased on 71/948 intervals (7.5%)
  oscRatio : increased on 104/948 intervals (11.0%)
  defect   : increased on 66/948 intervals (7.0%)
```

# Collapse Atlas run (E3), seed 20260718

## E3 metastability and spurious equilibria

```
E3: mass-imbalance relaxation (deterministic transport law)
  (a) deterministic halving flow-time T vs L/tau (hypothesis: log T linear in L/tau):
    L/tau= 2.00: flow-time to halve =      7.50
    L/tau= 2.50: flow-time to halve =     10.40
    L/tau= 2.94: flow-time to halve =     15.64
    L/tau= 3.57: flow-time to halve =     34.02
    L/tau= 4.17: flow-time to halve =     85.92
    L/tau= 5.00: flow-time to halve =    809.20
    L/tau= 5.88: flow-time to halve =    >12000*eta (stalled)
    fitted slope d(log T)/d(L/tau) = 1.519 (pure exponential law would be ~1)
  (b) spurious-equilibrium hunt at large separation (L/tau ~ 7):
    after 30000 steps: alpha=0.200 resid=1.10e-05
    Newton: success=False resid=7.91e-06 alpha=0.200 -> no exact spurious equilibrium
```

# Collapse Atlas run (E4), seed 20260718

## E4 basins

```
E4: basin fractions over (tau, N, spread); 60 inits each
  tau=1.0 N=8 spread=1.0: {'converged': 57, 'collapsed': 0, 'metastable': 0, 'wandering': 3}
  tau=1.0 N=8 spread=4.0: {'converged': 48, 'collapsed': 0, 'metastable': 0, 'wandering': 12}
  tau=1.0 N=32 spread=1.0: {'converged': 60, 'collapsed': 0, 'metastable': 0, 'wandering': 0}
  tau=1.0 N=32 spread=4.0: {'converged': 60, 'collapsed': 0, 'metastable': 0, 'wandering': 0}
  tau=2.5 N=8 spread=1.0: {'converged': 60, 'collapsed': 0, 'metastable': 0, 'wandering': 0}
  tau=2.5 N=8 spread=4.0: {'converged': 60, 'collapsed': 0, 'metastable': 0, 'wandering': 0}
  tau=2.5 N=32 spread=1.0: {'converged': 60, 'collapsed': 0, 'metastable': 0, 'wandering': 0}
  tau=2.5 N=32 spread=4.0: {'converged': 60, 'collapsed': 0, 'metastable': 0, 'wandering': 0}
  tau=6.0 N=8 spread=1.0: {'converged': 60, 'collapsed': 0, 'metastable': 0, 'wandering': 0}
  tau=6.0 N=8 spread=4.0: {'converged': 60, 'collapsed': 0, 'metastable': 0, 'wandering': 0}
  tau=6.0 N=32 spread=1.0: {'converged': 60, 'collapsed': 0, 'metastable': 0, 'wandering': 0}
  tau=6.0 N=32 spread=4.0: {'converged': 60, 'collapsed': 0, 'metastable': 0, 'wandering': 0}
```

---

# Pass 2 (audit-hardened), 2026-07-18

*Generated from `numerics/atlas_runs/*/` (manifests + raw CSVs inside).
Protocol: `CollapseAtlas.md` §Pass 2.  Audit: `CollapseAtlasAudit.md`.
Wording is evidence-calibrated: nothing here is "certified".*

## P2A — the fission law is exactly first-order (invariants pass)

Invariant suite (V(p,p)=0, translation equivariance, atom-split
invariance, weight normalization): all PASS.  The relative error of the
pair law `u ← u + η(I + Dm_p(c))u` equals the perturbation radius
(log–log slope 0.996 over radii `1e-2τ…1e-7τ`, roundoff below): the
pass-1 "uniform 1e-6 error" was the quadratic remainder, as predicted.

## P2B — the dichotomy conjecture, now sharp

Symmetry-quotiented adversarial search (differential evolution + 40
Nelder–Mead restarts per DE run, weight floor, exact-zero polishing,
three FD scales, singular values reported) in d = 2 and 3 drove σ to
`±1e-6` — but with `min singval(I+Dm) ≈ 1e-21`: the minimizer converges
to a **degenerate boundary**, not to a stable collapse.  Inspection
(`atlas_p2b_inspect.py`): the minimizing geometry places a dominant atom
of `p` (weight 0.86) at distance 4e-3 from the collapse point — σ → 0⁺
exactly in the "collapse onto an atom of p" limit, where `Dm → −I`.

**Refined conjecture (A2-sharp):** σ > 0 at every sink unless the sink
carries a locally dominant atom of `p`, and σ = 0 exactly in that limit —
i.e. *the only way point collapse approaches stability is by being where
`p` actually has a point mass*.  Matches the proven 1-d identity
`σ = (1/τ)·Cov_w(X, sgn X)`, which vanishes iff the tilted mass sits
entirely at the point.

## P2C — the metastability law replicates; the two-bandwidth rescue is real

* Exponential slope in `L/τ`: `c = 1.692` (L = 5) vs `c = 1.698` (L = 8)
  on scale-consistent families (σ = 0.25τ), zero censored cells —
  **`L/τ` controls the barrier**.  AIC mildly prefers a `x^a·e^{cx}`
  refinement on this narrow range (a < 0, likely overfit); exponential
  dominance is unambiguous.
* `η`-halving leaves `T/τ` unchanged (45.9 → 45.6): continuum behavior,
  not discretization.
* Fixed-absolute vs scale-consistent cluster width: negligible at the
  tested cell (46.5 vs 45.9) — the pass-1 conflation did not distort the law.
* **Two-bandwidth intervention** `V = ½(V_τ + V_{τ'})`:
  at `L/τ = 5`, single-bandwidth `T/τ = 679` drops to `21.5` (τ' = L/2)
  and `17.7` (τ' = L): **31–38× faster**.  The paper's multi-τ heuristic
  is a measured metastability rescue; a bandwidth-ladder design rule (G2)
  is now a quantitative target.

## P2D — the no-Lyapunov finding survives at the flow level

Analytic orbital derivatives `∇F·V` (closed-form atomic gradients):
MMD² strictly increases at 5.5% and energy distance at 6.8% of 2400
sampled states along trajectories — **genuine flow-level violations with
stored witnesses** (`p2d_witnesses.csv`), not Euler overshoot.  The A3
question is now firmly "find a cleverer functional or prove impossibility".

## P2E — the truth is stable for the population field; the mask breaks
## exact stationarity at finite N

Continuous generator at `q = p` (unmasked): `max Re eig < 0`, and the
Euler boundary from the spectrum matches bisection to 3 decimals
(`η* = 2.012τ` at τ=1, `0.808τ` at τ=2.5) — the step-size ceiling is now
predictable from the generator.  With the paper's self-mask, `q = p` is
**not stationary at finite N** and the linearization has expanding
directions (max Re eig ≈ +0.6) — consistent with the certified Obj-4
result that masking shifts the population target by O(1/N); the masked
dynamics equilibrates near, not at, `p`.  N-scaling of this shift is a
pass-3 item.

## Updated theorem targets

* **T1 (unchanged, Lean):** 1-d σ ≥ 0 from the certified derivative bound;
  strictness now has the right hypothesis: no dominant atom at the sink.
* **T2 (sharpened):** metastability bounds `e^{c₁L/τ} ≤ T ≤ e^{c₂L/τ}`
  with the measured `c ≈ 1.7`, plus the two-bandwidth acceleration bound.
* **T5 (new):** the masked-field equilibrium is `O(1/N)`-shifted from `p`
  and the unmasked Euler ceiling satisfies `η* = min −2Reλ/|λ|²` over the
  generator spectrum (numerically exact here).

---

# Pass 3 (numerics completion), 2026-07-18

*Generated from `numerics/atlas_runs/*P3*/`.  Figures in
`numerics/atlas_figs/` (metastability, splitting_1d, mask_shift).*

## P3A — the self-mask shift: law-level small, position-level not

Masked-field equilibria from exact p-representations (N = 6k particles):
the law-level gap decays fast — `ED²(p, q_eq)`: 2.37 (N=6) → 0.28 (12)
→ 0.0825 (24) → 0.0252 (48) → 0.0079 (96), roughly `N^{-1.7}` — but the
**maximum single-particle displacement plateaus at O(1)** (≈ 2.2): a
vanishing-mass stray particle remains far-displaced.  At N = 6 (one
particle per atom) the masked dynamics equilibrates *far* from p
(max-shift 6.7, ED² 2.4) and is genuinely stable there
(max Re eig = −0.12): **with very few particles per mode, the eye-mask
creates a stable wrong equilibrium.**  At N ≥ 24 the near-equilibrium is
marginal (max Re eig ≈ +0.003 with residual ~1e-4 — slow drift, not fully
settled at the step budget).  Caution flag for small-batch eye-masked
training; a longer-equilibration protocol and stray-particle tracking are
the remaining open items here.

## P3C — zero-finding coverage resolved (tested families)

1-d exhaustive bracketing over the atom hull (outer bound: the field
points inward outside the hull) reproduces the pass-1 census exactly
(3/3/1/1 zeros across τ; zero tangencies).  2-d multistart census is
stable from 250 → 1000 → 4000 seeds (3 zeros at both bandwidths).  The
pass-1 landscape was complete on these families.

## P3E — the bandwidth-ladder design curve

At `L/τ = 5` (single-bandwidth `T/τ = 679`), the mixture
`V = (1−β)V_τ + βV_{τ'}` gives:

| τ'/L | β=0.25 | β=0.5 |
|---|---|---|
| 0.25 | 291.8 | 153.2 |
| 0.5  | 66.0  | 21.5  |
| 1.0  | 59.2  | **17.7** |
| 2.0  | 72.6  | 21.9  |

**Design rule (measured):** the second bandwidth should sit at the
cluster-separation scale (`τ' ≈ L`, broad optimum from L/2 to 2L) with
substantial weight (β = 0.5 ≫ 0.25).  Too-small `τ'` (L/4) barely helps.
This is the quantitative seed of the G2 certified-bandwidth rule.

## P3B — the full basin matrix: collapse never happens

54 cells (P1 + P2, τ ∈ {0.2, 0.5, 1}·L, N ∈ {8, 32, 128}, spreads
{0.5, 2, 5}·L, 100 inits/cell — 50 for N = 128 — with 4× extension of
unresolved runs and per-cell quantization floors):

* **Zero collapse endpoints in ~4,750 generic runs** (Wilson 95% upper
  bounds ≤ 3.7% / 7.1% per cell), across every bandwidth, size, spread.
* **Collapse-biased initializations always escape**: 150 runs started
  *inside* a sink's fission zone (0.05τ jitter) ended metastable
  (N=8, τ=0.2L), law-accurate-moving (N=32, τ=0.2L), or fully converged
  (τ=L) — never collapsed.  The σ > 0 fission prediction holds in vivo
  at particle level, not just in linearization.
* Phase structure: at τ = 0.2L with N = 8, metastable mass imbalance
  (28–58%) plus slow unresolved runs dominate the failures; at larger N
  the picture shifts to `target_moving` (law-accurate by energy distance
  but above the strict stationarity tolerance at the step budget); at
  τ ≥ 0.5L essentially everything is target-accurate.
* Quantization floors behave as predicted (P2 with N = 8 cannot represent
  masses 0.3/0.7 exactly; floor 0.036 accounted for in the criterion).

**Atlas conclusion (evidence-calibrated):** across three passes, the only
failure modes of deterministic population drifting ever observed are
*slowness* — exponential mass-imbalance metastability at small bandwidth,
and slow final equilibration at large N — never point collapse, never a
stable wrong equilibrium (unmasked field).  Collapse exists as an exact
equilibrium but is dynamically irrelevant: unstable in linearization
(P2B, sharply), and never reached or retained in simulation (P3B).  The
practical design levers are the bandwidth ladder (P3E: τ' ≈ separation,
equal weight) and the step-size ceiling (P2E: η* from the generator).
