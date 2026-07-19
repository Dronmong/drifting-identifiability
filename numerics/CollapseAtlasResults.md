# Collapse Atlas — findings summary (2026-07-18)

*Curated conclusions; raw experiment logs appended below by
`collapse_atlas.py` (master seed 20260718).  Spec: `CollapseAtlas.md`.
The E3(b) residual-floor certification is in `atlas_e3b_check.py`.*

## Headline findings

1. **(E1) The spurious-equilibrium landscape is exactly the mean-shift mode
   structure.**  Zeros of `m_p` = sinks (one per resolved cluster) plus
   saddles/sources between them; sinks merge into one as `τ` passes the
   separation scale.  Lone-particle collapse candidates exist at every
   bandwidth.

2. **(E2) Collapse is always fission-unstable — the dichotomy conjecture
   survives adversarial attack.**  The pair-splitting linearization
   `u ← u + η·(I + Dm_p(c))u` is verified numerically (relative error at
   the perturbation scale).  The splitting index
   `σ(c) = max Re eig(I + Dm_p(c))` was positive at **every** sink of every
   family in d = 1, 2 (census minimum `+0.018`), and 24 adversarial
   Nelder–Mead searches over 4-atom 2-d configurations pinned to exact
   zeros could push it only to `+0.005`, never negative.
   *Conjecture (A2-dichotomy):* `σ ≥ 0` always, `> 0` unless degenerate;
   collapse can be made arbitrarily *slow* to escape but never trapping.
   **1-d theorem target (near-free in Lean):** `σ = 1 + m_p′(c) =
   (1/τ)·Cov_w(X, sgn X) ≥ 0` follows from the already-certified
   `laplaceMeanShiftRatioDeriv_add_one_nonneg`; strictness under a
   non-degeneracy hypothesis.

3. **(E3 + atlas_e3b_check) Wrong-mass states are NOT equilibria — but they
   are exponentially metastable.**  Deterministic halving time of a
   two-cluster mass imbalance grows like `exp(c·L/τ)` with fitted
   `c ≈ 1.5` over `L/τ ∈ [2, 5]`, and stalls beyond (`> 12000·η` at
   `L/τ = 5.9`).  At `L/τ ≈ 7`, least-squares certifies a **strictly
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
* **T3 (theory):** no-spurious-equilibria for finite atomic `q ≠ p` — is
  the certified positive residual floor a theorem?  (Connects to the
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
