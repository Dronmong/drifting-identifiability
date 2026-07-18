# The post-converse frontier (2026-07-18)

*Research pass conducted immediately after closing the general Euclidean
Laplace converse (`laplaceZeroDrift_identifies_euclidean`).  This document
derives **new** research directions from what the codebase now knows — not
formalizations of known results, but questions the completed program makes
newly askable and newly attackable.  Two of the sharpest claims below were
numerically verified before being written down (`numerics/frontier_screen.py`,
seed 20260718).*

---

## 0. What we actually own now (the asset inventory)

Reading the finished codebase as a whole, the assets are broader than "one
converse theorem":

1. **A two-function elliptic calculus for drift fields.**  Every statement
   about the Laplace drift was ultimately rewritten in the pair
   `(ψ_μ, Z_μ)` — the Matérn-3/2 companion potential and the normalizer —
   coupled by the *pointwise, classical, measure-free* identity
   `ψ − τ²Δψ = (n+1)τ²Z` and `∇ψ = D` (certified in
   `LaplaceDisplacementHessian.lean` / `LaplaceRadialFoundations.lean` for
   **arbitrary finite measures**, atoms included).  This is a genuinely
   unusual analytic setting: a nonlinear integral field (normalized mean
   shift) with an exact linear elliptic shadow.
2. **A global maximum principle for flow-monotone defects**
   (`LaplaceFoliationMaximum.lean`): decay at infinity + attained max +
   backward-flow strict growth with a *manifestly signed* Abel coefficient +
   speed bound ⟹ the defect vanishes.  The proof pattern only used:
   gradient alignment, the elliptic identity, `τ/e` field bound, Hessian
   bound `2`, and Grönwall.  Nothing about it is specific to probability,
   to the exact kernel, or (except compactness of sublevels) to finite
   dimension.
3. **A hard-won negative catalogue**: pointwise Jacobian sign control is
   *unboundedly false* in `n ≥ 2` (falsified §4.6(e), both forms); raw
   `‖V‖` does not control weak convergence for embedding fields;
   band-limited kernels fail identifiability (`FailureCases.lean`).
   Negative results of this precision are rare and delimit the design space
   for anything below.
4. **A finite-sample estimator theory** (SNIS/Algorithm-2 layer, Objectives
   4–7) with an honest certified-vs-observed sample-complexity ledger, whose
   dominant slack is known (deterministic SNIS denominator floor).
5. **An extension track already in motion** (Sinkhorn balancing depth,
   `SinkhornImplementation/PLAN.md`) that treats the paper's column softmax
   as half a Sinkhorn iteration.

The directions below are ordered by (novelty × leverage), not difficulty.

---

## Direction A — The dynamics of drifting: fixed points, collapse, and the
## support-restricted converse  *(the most important open question)*

**Insight from the code.**  The converse characterizes when the *field*
vanishes everywhere.  But the generative dynamics
`q ↦ (id + η·V(p,q))_# q` does not see the whole field: a measure `q` is
stationary iff `V(p,q) = 0` **on `supp q` only**.  Our theorem says nothing
about that weaker condition — and it is genuinely weaker:

> For `q = δ_c` we get `m_q ≡ 0` on `supp q`, so `δ_c` is a stationary point
> of the population dynamics iff `m_p(c) = 0`, i.e. `c` is a kernel-tilted
> barycenter of `p`.  Such `c` always exists.  **Collapse states are exact
> fixed points of drifting.**

So the honest uniqueness question for the *algorithm* (not the field) is:

* **A1 (support-restricted converse).**  If `V(p,q) = 0` on `supp q` and
  `supp q` is "large enough" (full support?  positive density?  merely
  non-atomic?), does `p = q` follow?  Where exactly is the threshold
  between the δ-collapse counterexample and the full-support theorem?
* **A2 (stability dichotomy).**  Conjecture: `p` is the unique *stable*
  equilibrium; every collapse state is linearly unstable for the population
  dynamics.  The linearization at `p` is exactly the response operator
  `T[h](x) = (1/Z_p)∫ k(x,y)(y − x − m_p(x)) dh(y)` whose injectivity we
  already screened numerically (rn_screen, σ_min bounds) — the missing part
  is the *spectral sign* analysis at `p` and at δ-collapses.
* **A3 (Lyapunov functional).**  The max principle is secretly a statement
  about monotonicity of `H` along the `D_q`-flow.  Is there a functional
  `𝓛(q)` decreasing along the population dynamics with `𝓛` minimized
  exactly at `p`?  Candidates to screen numerically: the smoothed-mass
  discrepancy `‖Z_p − Z_q‖`, the potential-ratio oscillation
  `osc(ψ_p/ψ_q)`, and the defect norm `‖Z_qψ_p − Z_pψ_q‖`.  A certified
  Lyapunov function would upgrade "unique equilibrium" to "global
  convergence" — the first end-to-end guarantee for a drifting-type
  generative method.

**Why this is the priority.**  It is the mathematical formalization of
*mode collapse* for this model class, it consumes the falsification
catalogue (pointwise sign control is dead, so any Lyapunov argument must be
integrated — exactly the lesson of RSI and of the max principle), and every
partial answer is publishable.  Numerics first: simulate the population
dynamics on atomic measures, measure basins of attraction of collapse vs
`p`, then attack A2's linearization.

---

## Direction B — A quantitative converse: metrize `‖V‖ → 0`

**Insight from the code.**  The proof of `H ≡ 0` is soft in exactly one
place: it uses `V ≡ 0` twice (alignment `D_p = R·D_q`, and the cancellation
`τ²·dR(D_q) = H`).  Both have natural `ε`-perturbed forms.  The rest of the
argument — decay, compact sublevels, backward growth, speed bound — is
already quantitative with explicit constants (`τ/e`, Hessian `2`,
`ψ_q ≤ τ²`, uniform Picard–Lindelöf time).

* **B1.**  Prove: `‖V(p,q)‖_∞ ≤ ε ⟹ ‖Z_p − Z_q‖_∞ ≤ C(τ, n, moduli)·ε^α`
  (a *stability* estimate in the smoothed metric — no deconvolution, so no
  ill-posedness).  The right unknown is not `R` (not differentiable off
  zero drift) but the **unnormalized defect** `K̃ = Z_qψ_p − Z_pψ_q` or,
  better, the pure-`ψ` formulation: using the elliptic identity, the entire
  drift theory is a statement about the quasilinear first-order system in
  `(ψ_p, ψ_q)` alone:
  `V = (n+1)τ²[∇ψ_p/(ψ_p − τ²Δψ_p) − ∇ψ_q/(ψ_q − τ²Δψ_q)]`.
* **B2.**  Downstream, decide in *which* metrics `‖Z_p − Z_q‖_∞` is a
  genuine discrepancy (it dominates a Matérn-RKHS MMD; it metrizes weak
  convergence on tight families).  This gives the drifting loss a certified
  surrogate: driving the empirical drift residual to `ε` provably brings
  the model within `δ(ε)` of the data law.  This is the "asymptotic V → 0"
  frontier that the 1-d program explicitly left open, now attackable with
  n-dimensional tools.

**First step.**  Numerically map `‖V‖_∞ ↦ ‖Z_p − Z_q‖_∞` over random
near-aligned pairs to guess the exponent `α` (linear?) before proving
anything.

---

## Direction C — The Matérn tower  *(numerically verified)*

**Insight from the code.**  Why did the Laplace kernel admit a companion
potential with an *exact* elliptic closure?  Working the general radial
computation backwards: a radial kernel `k(r)` has a companion `f` with
`∇f = D` iff `f' = −r·k` (always solvable); the pair closes elliptically,
`Δf = αf + βk`, iff the profile solves

```
r·k'' + (n + 1 + β)·k' − α·r·k = 0,
```

a modified-Bessel equation whose decaying solutions are exactly the
**Matérn family**.  Concretely (verified to finite-difference precision in
`numerics/frontier_screen.py`):

* Laplace = Matérn-1/2: companion is Matérn-3/2, `Δψ = ψ/τ² − (n+1)k` —
  the certified identity;
* Matérn-3/2 kernel `k = (1 + r/τ)e^{−r/τ}`: companion is the Matérn-5/2
  profile `e^{−r/τ}(r² + 3τr + 3τ²)`, with closure
  `Δf = f/τ² − (n+3)k`.

So the ladder `Matérn-ν → Matérn-(ν+1)` runs through all half-integers, with
the constant `β = −(n + 2ν + …)` shifting by 2 per rung.

* **C1 (Matérn converse).**  Port the maximum principle to every
  half-integer Matérn kernel: same skeleton (field bound, Hessian bound,
  companion positivity, first-moment decay, Abel coefficient = companion/τ²
  > 0), new constants.  Outcome: zero Matérn-ν drift identifies arbitrary
  probability measures in every finite dimension — covering the workhorse
  kernels of Gaussian-process modeling in one stroke, and (with ν → ∞
  degenerating to the already-closed Gaussian case) unifying both closed
  kernels under one mechanism.
* **C2 (classification).**  Prove the converse of the computation: the
  Matérn profiles are the **only** radial kernels whose drift theory admits
  a two-function elliptic closure.  This explains *a priori* why the paper's
  kernel was the tractable one, and tells us that any other kernel class
  needs a genuinely different mechanism (cf. Direction D).
* **C3 (completely monotone kernels).**  Cauchy/rational-quadratic kernels
  are mixtures of Laplace/Gaussian kernels over the bandwidth.  Mixtures
  wreck the *normalized* drift (normalizers don't mix), so the converse for
  them is genuinely open — a clean testbed for whether identifiability is a
  property of "heavy enough tails" or of the exact resolvent structure.

---

## Direction D — Sharpness: finite-range kernels are mass-blind
## *(counterexample verified; cheap theorem)*

**Insight from the code.**  The 1-d atomic converse worked by telescoping
*exponentially small* cross-cluster couplings.  What if the coupling is
exactly zero?  Take any compactly supported kernel (support radius `ρ`) and
two measures with the same cluster *shapes* but different cluster *masses*,
clusters separated by more than `2ρ`.  The normalized mean shift only sees
the local conditional law — the cluster masses cancel in `D/Z` — and probes
in the dead zone have `Z = 0` (the Lean `(0)⁻¹ • D = 0` convention makes the
field literally `0` there).  Verified: `max‖V‖ = 8×10⁻¹⁷` for Wendland with
cluster masses `(0.5, 0.5)` vs `(0.2, 0.8)`, while the Laplace control sees
`‖V‖ ≈ 3.2`.

* **D1.**  Formalize: for every compactly supported kernel, the zero-drift
  converse **fails**, with the explicit two-cluster witness.  A few hundred
  lines; instantly delineates the theory.  Corollary of interest to
  practitioners: Epanechnikov/Wendland mean-shift-style generative losses
  cannot identify multi-modal targets — **the Laplace tail is
  load-bearing**, and "characteristic kernel" (an MMD notion) is *not* the
  right dividing line for normalized drifts.
* **D2 (the true dividing line).**  Define *drift-characteristic* and
  classify: strict positivity is necessary (D1); is it sufficient?  The
  candidate boundary cases are super-exponentially decaying kernels
  (Gaussian: true, known) vs sub-exponential vs polynomial tails (open,
  C3).  A clean conjecture to hunt numerically: the converse holds for
  every strictly positive kernel with `−log k(r) = o(r²)`… or fails for
  some?  Either answer is a paper-grade result.

---

## Direction E — Local-to-global: how much of the field pins the measure?

Practice never evaluates `V` off the data/model manifold.  Between the
support-restricted question (A1) and the global theorem lies:

* **E1.**  `V = 0` on an *open set* `U` ⟹ `p = q`?  The flow machinery
  localizes (alignment and the Abel ODE hold on `U`), but the max principle
  is global.  The kernel kink at collision breaks naive analytic
  continuation, yet `Z` is analytic off the supports — there may be a
  unique-continuation route for the elliptic pair.  A numeric counterexample
  search (optimize atomic `p ≠ q` to zero the drift on a ball) is the right
  first move; the linearized-injectivity screens suggest local rigidity,
  which would make a *positive* answer plausible.
* **E2.**  `V = 0` outside a compact set (far-field only) ⟹ `p = q`?  The
  L1 far-field layer already extracts tilted centroids per direction; the
  question is whether the full tilt transform is forced, i.e. whether the
  far field alone is a complete set of observables.

---

## Direction F — Infinite dimensions via Ekeland

The maximum principle uses finite-dimensionality exactly once: compactness
of the sublevel sets of `H` (to attain the sup).  The Ekeland variational
principle supplies `ε`-maxima on complete spaces with *no compactness*, and
the backward-flow growth argument only needs an `ε`-max with quantified
slack — the growth factor `e^{δT}` beats slack `ε` for `ε` small.

* **F1.**  Re-run the max principle on a separable Hilbert space via
  Ekeland: conclude `H ≡ 0` and ratio constancy for arbitrary probability
  measures on `ℓ²`.  (The flow layer — Picard–Lindelöf, Grönwall, speed —
  is already dimension-free in the Lean development; the variable blocks
  say `[CompleteSpace E]`, not `[FiniteDimensional ℝ E]`, for most of it.)
* **F2.**  The remaining obstruction is smoothing injectivity
  (`Z_p = c·Z_q ⟹ p = c·q`) in infinite dimensions — finite-dimensional
  Fourier dies, but the Bernstein-subordination representation of the
  Laplace kernel as a Gaussian mixture survives, and Gaussian smoothing
  injectivity on Hilbert spaces is approachable via cylindrical
  projections and the *already-certified* finite-dimensional theorem
  applied to every finite-dimensional marginal.  Outcome: identifiability
  for function-space generative models — beyond anything in the current
  literature on drift-based methods.

---

## Direction G — The bandwidth dial: score matching at `τ → 0`, and what
## `τ` interpolates

For smooth densities, the tilted mean satisfies
`m_p(x) = τ·c₁·(∇p/p)(x)·τ + o(τ²)`-type expansions (the kernel localizes),
so `V_τ/τ² → c·(∇log p − ∇log q)`: **drifting interpolates between score
matching (`τ → 0`) and global attraction–repulsion (`τ` large)**.

* **G1.**  Certify the expansion with explicit remainder for `C¹` densities:
  `‖V_τ − c τ²(∇log p − ∇log q)‖ ≤ C(p,q) τ³` on compacts.  This is the
  formal bridge between the drifting family and diffusion/score-based
  models, and makes the multi-`τ` design of the paper (three bandwidths)
  interpretable as multiscale score estimation — with the certified ESS
  numerics (`numerics/RESULTS.md`) already showing exactly this regime
  split empirically.
* **G2.**  Exploit it: identifiability at a *single* small `τ` already
  holds (our theorem is per-`τ`); the interesting quantitative question is
  which `τ` optimizes the stability constant of Direction B — a certified
  answer to "how should bandwidth be chosen," which the paper sets by grid
  search.

---

## Direction H — Companion-potential drifting: a new algorithm with
## gradient structure

**Insight from the code.**  The mean-shift drift `V` is not a gradient
field (the normalizers break exactness), which is why nothing like an
energy-descent analysis exists for Algorithm 2.  But the closure theory
says the *potential ratio* `G = ψ_p/ψ_q` carries the same information:
`G ≡ const ⟺ p = q` (apply `(1 − τ²Δ)` to `ψ_p = c·ψ_q` to get
`Z_p = c·Z_q`, then the certified endgame).  Define the **companion drift**

```
V̂(p,q) := ∇(ψ_p/ψ_q) = (Z_q-weighted curl-free field),
```

which is (i) *exactly* a gradient field, so the induced population dynamics
is a genuine gradient flow with `osc(ψ_p/ψ_q)` as a canonical Lyapunov
candidate — cycling and (plausibly) some collapse modes are structurally
excluded; (ii) sample-computable exactly like mean shift, since `ψ` is just
kernel smoothing with the Matérn-3/2 profile `τ(r+τ)e^{−r/τ}` (heavier
tails than the Laplace kernel — the `Z → 0` far-field instability of the
normalized drift is *milder* for `ψ`); (iii) already fully covered by our
identifiability machinery — the equilibrium theory comes for free.

* **H1.**  Numerics: implement companion drifting in `driftlab.py` next to
  Algorithm 2; compare collapse behavior, far-field behavior, and sample
  complexity at the paper's operating point.
* **H2.**  If the numerics are favorable: certify the Lyapunov property at
  the population level (this is Direction A3 for a field *designed* to make
  A3 provable), and port the SNIS estimator layer (the `ψ`-smoother has the
  same self-normalized structure).

This is the clearest opportunity to "make something new and powerful": a
drifting variant whose *design* is dictated by the theorems, rather than
theorems chasing the design.

---

## Existing tracks this slots above (not superseded)

* **Sinkhorn balancing depth** (`SinkhornImplementation/PLAN.md`) — the
  entropic-OT reading of Algorithm 2's affinity; complementary to H (both
  are principled field modifications; they compose).
* **Bernstein SNIS denominator bound** — the identified dominant slack in
  the certified sample-complexity ledger; still the highest-value
  *incremental* theorem in the estimator layer.

## Suggested sequencing

| Rank | Direction | First deliverable | Mode |
|---|---|---|---|
| 1 | A (dynamics/collapse) | population-dynamics simulator + fixed-point/stability atlas | numerics → theory |
| 2 | H (companion drifting) | `driftlab` head-to-head vs Algorithm 2 | numerics → design → Lean |
| 3 | D (finite-range sharpness) | two-cluster counterexample theorem | Lean (cheap) |
| 4 | C (Matérn tower) | Matérn-3/2 converse port | Lean (mechanical, medium) |
| 5 | B (quantitative converse) | ε-perturbed max principle | theory (hard, high value) |
| 6 | E, F, G | counterexample screens / Ekeland port / τ-expansion | mixed |

D and C are the natural "next Lean milestones" (one cheap, one mechanical);
A and H are where genuinely new science lives; B is the theorem that would
matter most to practitioners if it lands.
