# The Collapse Atlas: fixed points and stability of drifting dynamics

*Procedure specification (2026-07-18).  Implements Direction A (A1/A2/A3) of
`DriftingIdentifiability/PostConverseFrontier.md` — the top-ranked open
question after the literature review.  Companion code:
`numerics/collapse_atlas.py`; results: `numerics/CollapseAtlasResults.md`.*

---

## 1. What we are mapping, and why it is open

The completed converse says: `V(p,q) ≡ 0` **everywhere** ⟹ `p = q`.  The
*algorithm*, however, evolves a particle measure by

```
x_j  ←  x_j + η · V(p, q)(x_j),        q = Σ_j w_j δ_{x_j},
```

so a state is stationary iff `V(p,q) = 0` **on `supp q` only**.  This is
strictly weaker: we showed `q = δ_c` is an exact fixed point whenever
`m_p(c) = 0` (a kernel-tilted barycenter/mode of `p`), and the verified
literature (Gradient Flow Drifting 2603.10592, WGF interpretation
2605.05118, finite-particle rates 2605.22795) does **not** classify these
support-restricted equilibria or their stability for the original Laplace
displacement field.  The atlas is a systematic numerical map of:

* **A1** — which configurations are (exactly or nearly) stationary;
* **A2** — which of them are dynamically stable, in which topology, and
  against which perturbation classes;
* **A3** — which candidate functionals actually decrease along the flow.

Every experiment below states a falsifiable hypothesis and the theorem it
would feed.

---

## 2. Objects (exact population formulas, no sampling noise)

All measures are atomic and all fields are computed **exactly** in float64 —
this is population-level dynamics, not the SNIS estimator (that comparison
comes later; the estimator layer already has its own certified analysis).

* Kernel: `k(x,y) = exp(−‖x−y‖₂/τ)` (the paper's ℓ² Laplace kernel).
* Normalized mean shift of an atomic measure `μ = Σ a_i δ_{y_i}`:
  `m_μ(x) = Σ a_i k(x,y_i)(y_i − x) / Σ a_i k(x,y_i)`.
* Drift `V(p,q)(x) = m_p(x) − m_q(x)`.
* Population update: simultaneous `x_j ← x_j + η·V(p,q)(x_j)` with weights
  `w_j` **fixed** (mass moves only by particle transport — this is what the
  algorithm does, and it is what makes cluster-mass relaxation slow).
* Self-interaction: by default the particle's own mass is **included** in
  `m_q` (honest population field; `k(x,x)=1` contributes only to the
  normalizer).  A `mask_self` flag reproduces the paper's eye-mask variant;
  E5 checks the difference is O(1/N) as the certified Obj-4 analysis
  predicts.
* Step size: default `η = 0.1·τ` (flow-like regime); E5 varies `η`
  explicitly to separate map-instabilities from flow-instabilities.

Diagnostics computed against a fixed probe grid `G` (dense in the hull of
`supp p ∪ supp q`, plus a far ring):

* `‖V‖_G = max_{x∈G} ‖V(p,q)(x)‖` (field residual off-support),
* `resid = max_j ‖V(x_j)‖` (on-support residual = stationarity check),
* energy distance `ED(p,q)` (a genuine metric on laws, cheap and exact for
  atoms), and in 1-d exact `W₂`,
* smoothed-mass gap `‖Z_p − Z_q‖_G`,
* Laplace-RKHS `MMD²` (note: for this kernel `‖Z_p − Z_q‖_∞ ≤ MMD`, the
  review's corrected direction),
* companion objects `ψ_μ(x) = Σ a_i τ(‖x−y_i‖+τ)e^{−‖x−y_i‖/τ}`, the ratio
  oscillation `osc_G(ψ_p/ψ_q)` and defect `‖Z_qψ_p − Z_pψ_q‖_G`.

---

## 3. Experiments

### E1 — Fixed-point census (the equilibrium landscape)

**Protocol.**  For a family of targets `p` (1-d and 2-d):

* `P1`: two clusters, separation `L ∈ {2, 5, 10}·τ`, equal masses;
* `P2`: two clusters, masses `(0.3, 0.7)`;
* `P3`: three collinear clusters (the rn_screen4 adversarial geometry);
* `P4`: an asymmetric 5-atom cloud (generic case),

find **all** zeros of `m_p` by dense grid scan + Newton polish (`m_p` is
smooth off atoms; at atoms use the certified fact that `m_p` is still C¹).
Classify each zero `c` by the eigenvalues of `Dm_p(c)` (finite differences,
central, `h = 1e-5·τ`):

* mean-shift sink (all eigenvalues of `Dm_p` in the left half plane under
  the flow `ẋ = m_p`) — these are the *modes* where a lone particle parks;
* saddles / sources.

**Hypothesis H1.**  Zeros of `m_p` ≈ modes+saddles of the τ-smoothed
density; the number of sinks equals the number of well-separated clusters
for `L ≳ 3τ` and collapses to 1 as `τ` grows past the separation scale.

**Feeds.**  The candidate set for every later experiment; the τ-dependence
quantifies which bandwidths even *have* spurious single-point equilibria.

### E2 — The splitting index (is collapse always escapable?)

The heart of A2.  A lone particle at a sink `c` is *stable as a lone
particle* (mean-shift ascent).  Escape must come from **fission**: split
`δ_c → ½δ_{c+u} + ½δ_{c−u}`.  First-order prediction (derived by hand, to
be verified): the pair separation evolves as `u ← u + η·(I + Dm_p(c))u`,
because the twin's repulsion contributes exactly `+u` at leading order.

**Protocol.**
1. Verify the linearization: simulate tiny pairs (`‖u‖ = 1e-6`) at each
   sink from E1 and regress the observed growth matrix against
   `I + Dm_p(c)`.
2. Compute the **splitting index** `σ(c) := max Re eig (I + Dm_p(c))` at
   every sink of every family, across `τ`.
3. 1-d anchor: the certified bound `m′ + 1 ≥ 0` (the proven
   `laplaceMeanShiftRatioDeriv_add_one_nonneg`) predicts `σ ≥ 0` always;
   check whether `σ = 0` is ever approached (degeneracy).
4. **Adversarial search (2-d/3-d):** minimize `σ(c)` over atom
   configurations of `p` subject to `m_p(c) = 0` (penalty method,
   Nelder–Mead + random restarts, warm-started from the rn_screen4
   families that made `sym(I + Dm)` unboundedly negative at non-zero
   points).  The falsified §4.6(e) says `I + Dm` can be very negative
   *somewhere*; the question is whether it can be negative **at a zero of
   `m_p`**.

**Hypothesis H2 (the dichotomy conjecture).**  `σ(c) > 0` at every sink of
every non-degenerate `p` — collapse is always linearly escapable by
fission.  **Either outcome is a result**: confirmation across adversarial
search ⟹ the "collapse is unstable" conjecture with strong evidence and a
1-d proof sketch already in hand; a counterexample ⟹ *stable collapse
exists* for Laplace drifting — a sharp warning with practical consequences
and an immediate theorem target.

### E3 — Metastability of mass imbalance (the e^{L/τ} law)

Even when collapse is escapable, *partial* collapse (right clusters, wrong
masses) should be nearly stationary: cross-cluster coupling is `e^{−L/τ}`,
and the mass-blindness theorem (Direction D) says it is exactly stationary
at `e^{−L/τ} = 0`.

**Protocol.**  Two-cluster `p` (masses ½/½, separation `L`), initialize `q`
with correct cluster shapes but particle counts `(αN, (1−α)N)`,
`α ∈ {0.2, 0.35}`.  Run the dynamics; record the cluster-mass trajectory
`α_t` (count particles by nearest cluster) and the relaxation time
`T(L, τ, η) := min{t : |α_t − ½| < ½|α₀ − ½|}`.

**Hypothesis H3.**  `log(η·T) ≈ L/τ + O(log)` — exponential metastability
in the separation-to-bandwidth ratio.  Additionally: a two-bandwidth field
`V = ½(V_{τ} + V_{τ'})` with `τ' ≈ L` collapses the barrier (this is the
certified-numerics ESS story from `RESULTS.md`, now measured as a
*dynamical* claim: the paper's multi-τ heuristic becomes a metastability
theorem target).

**Feeds.**  A2's "stability topology" refinement (partial collapse is not
an equilibrium but is `e^{L/τ}`-slow — indistinguishable in practice), and
a quantitative design rule for bandwidth ladders (G2 tie-in).

### E4 — Basins of attraction

**Protocol.**  For `P1`/`P2` at `τ ∈ {0.2, 0.5, 1}·L`: 200 random particle
initializations (N ∈ {8, 32, 128}, i.i.d. Gaussian with spread `s ∈
{0.5, 2, 5}·L` around the barycenter).  Run to `T_max`; classify endpoint:

* `converged`: energy distance to `p` below tolerance and residual small;
* `collapsed`: all particles within `ε`-ball (or all at one cluster);
* `metastable`: small residual but wrong cluster masses;
* `wandering`: none of the above at `T_max`.

Report the phase fractions over `(τ, N, s)`.

**Hypothesis H4.**  Collapse basin fraction grows with `τ/L` (single big
mode) and shrinks with `N`; for `τ ≪ L` the dominant failure is
`metastable`, not `collapsed` — i.e. **the practical failure mode of
drifting is mass imbalance, not point collapse**.

### E5 — Stability at the truth, step size, and the mask

**Protocol.**  Set `q = p` exactly (particles at the atoms with the right
weights).  Assemble the full `Nd × Nd` Jacobian `J` of one update
(finite differences).  Spectral radius `ρ(J)` vs `η/τ`; the threshold
`η*(τ)` where `ρ = 1`.  Repeat with `mask_self = true` to measure the
mask's shift of `η*` (predicted `O(1/N)`).

**Hypothesis H5.**  `p` is asymptotically stable for `η < η*` with
`η* = c·τ` (the field's Lipschitz constant is `2` in units where speeds
are `τ/e`, so `c` should be near `1`); no eigenvalue crosses through
instability *before* the generic `η`-overshoot — i.e. the truth is never
dynamically repelling in the flow limit.

### E6 — Lyapunov candidates screen (A3, and the impossibility angle)

**Protocol.**  Along every E3/E4 trajectory, record at each step:
`ED(p,q_t)`, `W₂` (1-d), `‖Z_p − Z_{q_t}‖_G`, `MMD²`, `osc_G(ψ_p/ψ_q)`,
`‖Z_qψ_p − Z_pψ_q‖_G`, and the on-support residual.  For each candidate:
fraction of steps that increase it (per trajectory and aggregate), and
where in state space the violations concentrate.

**Hypothesis H6.**  No candidate is globally monotone (consistent with
Balasubramanian's non-conservative residual for this field); violations
concentrate near saddle crossings and during cluster-mass transport.  A
candidate that *is* empirically monotone would immediately become the A3
theorem target; universal failure feeds the "no Lyapunov functional of
this class" impossibility direction instead.

---

## 4. Deliverables

1. `collapse_atlas.py` — all experiments, deterministic seeds, one entry
   point per experiment, modest runtimes (minutes each).
2. `CollapseAtlasResults.md` — auto-appended numeric summaries + the
   phase-fraction tables; figures (`numerics/atlas_figs/`) for the
   splitting-index landscape, metastability scaling, and basin maps.
3. A short **theorem-target list** extracted from the outcomes, e.g.:
   * 1-d fission instability of collapse from the certified `m′ + 1 ≥ 0`
     (if H2 confirms, this is a near-free Lean theorem on top of the
     existing one-sided derivative layer);
   * the metastability exponent as a conjecture with precise constants;
   * whichever Lyapunov candidate survives E6 (or the impossibility
     statement if none does).

## 5. Reproducibility

```
uv run --with numpy --with scipy --with matplotlib python numerics/collapse_atlas.py [E1|E2|E3|E4|E5|E6|all]
```

Master seed 20260718; every experiment reseeds `default_rng(20260718 + k)`.
All dynamics are deterministic given the initialization; randomness enters
only through initial conditions and adversarial-search restarts.
