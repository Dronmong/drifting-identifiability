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

---

## Literature-aware assessment and revised research triage (2026-07-18)

This section audits Directions A--H against literature available as of
2026-07-18.  It does not replace the proposals above: it records which parts
remain exciting and plausibly new, which parts have recently been answered,
and where the proposed mathematical route needs tightening.

The central conclusion is that the preceding sections are a strong internal
idea map, but not yet a reliable novelty map.  Several papers from March--May
2026 overlap substantially with Directions B, C, G, and H.  The best remaining
research opportunities are the support-restricted dynamics in A, an explicit
quantitative refinement of B, finite-range sharpness and the broader
drift-characteristic classification in D/C3, and the local-to-global questions
in E.

### Summary ratings

| Direction | Excitement | Promise | Current literature status |
|---|---:|---:|---|
| A -- dynamics/collapse | 9/10 | 7/10 | Exact support-restricted problem still appears open |
| B -- quantitative converse | 8/10 | 6/10 | Qualitative stability largely answered; explicit rates remain |
| C -- Matérn tower | 3/10 overall | 9/10 technically | C1/C2 already solved; C3 remains interesting |
| D -- finite-range failure | 7/10 | 9/10 | Cheap, solid, and apparently not stated in this exact form |
| E -- local-to-global | 9/10 | 5/10 | Largely open for Laplace drift and arbitrary measures |
| F -- infinite dimensions | 8/10 | 3/10 as proposed | Smoothing injectivity is known; the proposed PDE route has a major gap |
| G -- bandwidth/score limit | 5/10 | 8/10 | G1 is mostly classical; G2 remains valuable |
| H -- companion drifting | 5/10 scientifically | 8/10 experimentally | Essentially already proposed in recent literature |

These scores measure research value, not formalization value.  A theorem may
still be worth formalizing even when its informal mathematics has appeared,
but it should not then be presented as a new mathematical discovery.

### Direction A -- dynamics, fixed points, and collapse

This is the strongest direction in the document.

Recent papers establish gradient-flow interpretations for Gaussian or
redesigned drifting fields, but generally work with smooth positive densities
or require a field to vanish globally.  They do not settle the singular
equilibrium condition

```text
V(p,q) = 0 only on supp(q).
```

That distinction is precisely where atomic collapse lives.  In particular,
the recent *Gradient Flow Drifting* and *On the Wasserstein Gradient Flow
Interpretation of Drifting Models* papers do not appear to classify all
support-restricted equilibria of the original Laplace displacement field:

* J. Cao, Z. Wei, and Y. Liu,
  [Gradient Flow Drifting](https://arxiv.org/abs/2603.10592).
* A. Gretton et al.,
  [On the Wasserstein Gradient Flow Interpretation of Drifting Models](https://arxiv.org/abs/2605.05118).

Several refinements are needed before A becomes a precise program:

* Full or merely dense support is an easy positive case: continuity extends
  zero drift from `supp q` to the whole space, after which the completed
  converse applies.
* A positive density is sufficient only when it really entails full support.
  Non-atomicity by itself is much too weak; a non-atomic law may be supported
  on a curve, surface, or other thin set.
* Stability must specify a topology and a perturbation class.  A collapsed
  Dirac may be stable against translations within the Dirac manifold while
  unstable against particle-splitting perturbations.
* Consequently, "every collapse state is linearly unstable" should remain a
  conjecture until the state space and linearized modes are made explicit.

A3 is more crowded than the original discussion suggests.  Gradient-flow and
Lyapunov results now exist for conservative KDE-gradient variants.  Meanwhile,
the finite-particle paper below explicitly distinguishes the original Laplace
displacement field as non-conservative and identifies an unavoidable residual:

* K. Balasubramanian,
  [Finite-Particle Convergence Rates for Conservative and Non-Conservative Drifting Models](https://arxiv.org/abs/2605.22795).

For the original Laplace field, an impossibility theorem or a precise
"no functional of a specified class can generate this field" result may be
more plausible, and more novel, than a conventional Lyapunov theorem.

**Assessment.**  Pursue A1 and A2 first.  They are the clearest genuinely open
scientific questions exposed by the formal converse.

### Direction B -- a quantitative converse

This remains valuable, but it must be reformulated around what Lee and Chun
have already proved.  Their June 2026 revision establishes that:

* convergence of the field alone does not force weak convergence, because mass
  may escape to infinity;
* tightness removes this obstruction and restores weak stability;
* without tightness, every `C₀`-vague cluster point lies on the defect ray
  `{c p | 0 <= c <= 1}`;
* one additional scalar observable can detect the missing mass.

See H. G. Lee and H. Chun,
[Identifiability and Stability of Generative Drifting with Companion-Elliptic
Kernel Families](https://arxiv.org/abs/2604.24196).  The general relationship
between characteristic kernel discrepancies and weak convergence is also well
developed; see C.-J. Simon-Gabriel et al.,
[Metrizing Weak Convergence with Maximum Mean
Discrepancies](https://arxiv.org/abs/2006.09268).

Thus much of B2 and the qualitative/sequential version of B1 is already
answered.  The genuinely useful remaining theorem is an explicit,
non-asymptotic estimate of the form

```text
driftNorm(p,q) <= epsilon  ->  distance(p,q) <= C * epsilon^alpha
```

under a quantitatively stated anti-escape condition, for example:

* common bounded support;
* an explicit uniform tightness modulus;
* a uniform moment/tail bound strong enough to imply that modulus; or
* the Lee--Chun overlap scalar bounded below.

Without such a restriction, no uniform modulus can hold because of the
mass-escape counterexamples.

One sentence in the original B2 proposal also needs a kernel-by-kernel check.
For the same RKHS kernel with `k(x,x) = 1`, the reproducing property gives the
automatic inequality

```text
sup_x |Z_p(x) - Z_q(x)| <= MMD_k(p,q),
```

not the reverse.  Any claim that the sup normalizer discrepancy dominates an
MMD needs additional assumptions or a deliberately different Matérn kernel.

**Assessment.**  A sharp explicit rate under an overlap/tightness certificate
would still be a strong result and would go materially beyond the existing
sequential theorem.

### Direction C -- the Matérn tower

C1 and C2 are already answered almost verbatim by Lee and Chun.  Their current
paper proves identifiability for companion-elliptic kernels and classifies that
class as the Gaussian and Matérn families with positive smoothness.  This is
strictly broader than the half-integer ladder proposed above.

Therefore:

* **C1 (Matérn converse): already answered.**
* **C2 (classification of two-function elliptic closure): already answered.**
* **C3 (kernels outside the companion-elliptic class): still interesting.**

Formalizing C1/C2 could provide independent machine-checked verification, but
would no longer constitute a new informal mathematical result.  Also, the
Gaussian `nu -> infinity` limit requires the correct bandwidth rescaling and
does not by itself transfer a maximum-principle proof.

C3 is the useful salvage.  Rational-quadratic, Cauchy, inverse-multiquadric,
and scale-mixture kernels are natural tests because characteristicness of their
unnormalized embeddings does not immediately control the nonlinear normalized
displacement field.

**Assessment.**  Do not make C1 or C2 the next research objective.  Fold C3
into the more general drift-characteristic classification proposed in D2.

### Direction D -- finite-range kernels are mass-blind

D1 is probably the best immediate theorem in the document.  It is clean,
inexpensive, and sharply separates normalized displacement drift from ordinary
kernel-mean embedding theory.  Compactly supported kernels may still be
characteristic in the usual RKHS sense, so an explicit mass-blindness theorem
would prove that characteristicness is not the correct dividing line for
normalized drift.  For background on characteristic kernel embeddings, see
B. K. Sriperumbudur et al.,
[Hilbert Space Embeddings and Metrics on Probability
Measures](https://www.jmlr.org/papers/v11/sriperumbudur10a.html).

The theorem should not literally quantify over every pathological compactly
supported function.  A safe statement should assume a standard nonnegative
finite-range radial kernel, nontrivial near zero (for example `k(0) > 0`), and
make the zero-normalizer convention explicit.

D2 is more exciting and much harder.  The claim "strict positivity is
necessary" does not follow from D1.  D1 proves failure when exact zeros create
disconnected interaction components.  A kernel with isolated zeros might still
be drift-characteristic.  The true condition may be global interaction
connectivity rather than pointwise strict positivity.  Likewise, the proposed
tail boundary `-log k(r) = o(r^2)` is a conjectural numerical target, not yet a
literature-backed classification.

**Assessment.**  Prove D1 soon.  Then use it to formulate D2 carefully, with
counterexample searches preceding any general conjecture.

### Direction E -- local-to-global identifiability

This is the highest-risk and highest-theoretical-upside direction.

For Gaussian kernels, the open-set version is close to classical: smoothed
densities are analytic, so local score agreement can propagate globally.  The
genuinely new case is the nonsmooth Euclidean Laplace kernel with arbitrary
measures and possible collisions with the supports.

The problem should be split into four regimes:

1. `U` is disjoint from both supports, where analytic/elliptic continuation is
   most plausible.
2. `U` intersects one or both supports, where the kernel kink is central.
3. Far-field equality for compactly supported measures.
4. Far-field equality for arbitrary heavy-tailed measures.

The statement "Z is analytic off the supports" must be used locally: one needs
a positive distance from a point to the closed support before differentiating
under the integral with uniform analytic control.

A targeted literature search did not locate a theorem settling these exact
Laplace normalized-drift questions.  That is not a proof of novelty, but it
makes E a credible frontier.  Atomic counterexample optimization remains the
right first step before a long formal development.

**Assessment.**  Very exciting, but only moderately promising until the
atomic and finite-support cases survive systematic falsification.

### Direction F -- infinite dimensions

The endpoint is exciting, but the proposed route is too optimistic.

F2, injectivity of Laplace smoothing on separable Hilbert spaces, is essentially
already available through characteristic radial-kernel theory.  In particular,
Hilbert-space Laplace kernels belong to known integrally strictly
positive-definite classes; see J. Ziegel, D. Ginsbourger, and L. Duembgen,
[Characteristic Kernels on Hilbert Spaces, Banach Spaces, and on Sets of
Measures](https://arxiv.org/abs/2206.07588).

The real obstruction is F1.  Finite dimensionality was not used only to attain
a maximum.  The completed proof also uses the dimension-dependent elliptic
identity

```text
Delta psi = psi / tau^2 - (n+1) Z.
```

There is no straightforward infinite-dimensional trace Laplacian with this
identity.  Ekeland replaces compact maximum attainment but does not replace the
elliptic closure driving the maximum principle.  Finite-dimensional projection
is also not automatic: smoothing by `exp(-||x-y||/tau)` does not commute with
projecting the measure, because the omitted orthogonal coordinates still
contribute to the full norm.

**Assessment.**  A Hilbert-space normalized-drift converse would be genuinely
new, but it needs a substantially different, non-Laplacian mechanism.  The
Ekeland-only route has low probability of success as written.

### Direction G -- bandwidth and the score limit

The qualitative connection is classical.  Mean shift has been understood as a
normalized KDE-gradient or shadow-kernel gradient since at least Y. Cheng,
[Mean Shift, Mode Seeking, and
Clustering](https://proceedings.mlr.press/r0/cheng95a/cheng95a.pdf).  Recent
drifting papers now make the Gaussian score/Wasserstein connection explicit.

For the unsquared Laplace displacement kernel, a rigorous small-bandwidth
expansion with explicit constants still has value, but is an incremental
refinement rather than a new conceptual discovery.

The regularity proposed in G1 is too weak for its stated remainder:

* `C^1` regularity generally gives an `o(tau^2)` expansion.
* A uniform `O(tau^3)` remainder normally needs a locally Lipschitz gradient
  or `C^2`-type control.
* A score-ratio bound requires the densities to be bounded away from zero on
  the compact region under consideration.
* Uniform control also needs explicit tail assumptions when the expansion is
  derived from a noncompact kernel integral.

G2 is more interesting than G1: optimize bandwidth against the quantitative
stability modulus, estimator variance, ESS, and finite-sample bias.  That would
turn the paper's grid search into a certified design rule.

**Assessment.**  G1 is feasible supporting work.  G2 is the actual research
contribution.

### Direction H -- companion-potential drifting

This field is implementable, but it is no longer a new algorithmic family.
After normalizing the companion smoother, the proposal

```text
Vhat(p,q) = grad(psi_p / psi_q)
```

is a smoothed density-ratio-gradient velocity.  It is the reverse-KL row of the
KDE Wasserstein-flow construction in *Gradient Flow Drifting*.  The broader
paper by M. Esteban-Casadevall et al.,
[Kernel-Gradient Drifting Models](https://arxiv.org/abs/2605.10727), similarly
replaces Euclidean displacement directions with kernel gradients and obtains
smoothed-KL descent and characteristic-kernel identifiability.

Three claims in the original H discussion need correction:

* Being a spatial gradient does not by itself prove that a `q`-dependent field
  is the Wasserstein gradient of a functional.  Here that conclusion can be
  justified through the reverse-KL KDE construction, but it is a separate
  variation-in-measure calculation.
* Collapse is not structurally eliminated.  For `q = delta_c`, if `c` is a
  critical point of `psi_p`, then `grad(psi_p / psi_q)` still vanishes at `c`
  because the centered `psi_q` has zero gradient there.  Stability may change,
  but the fixed point remains.
* The established natural Lyapunov functional is a smoothed reverse KL, not
  automatically `osc(psi_p / psi_q)`.  The oscillation would need its own
  monotonicity proof.

**Assessment.**  H remains worthwhile as an empirical benchmark and as a
formally verified special case, but it should not be the project's flagship
novelty claim.

### Existing incremental tracks

The Sinkhorn track is also substantially occupied by current literature:

* P. He et al.,
  [Sinkhorn-Drifting Generative Models](https://arxiv.org/abs/2603.12366).
* J. Han et al.,
  [One-Step Generative Modeling via Wasserstein Gradient
  Flows](https://arxiv.org/abs/2605.11755).

The codebase's Sinkhorn formalization and estimator analysis can still be
valuable, especially where they verify exact implementation semantics or prove
finite-depth concentration, but the underlying algorithm should not be claimed
as new.  The Bernstein SNIS denominator improvement remains a useful technical
increment rather than a standalone scientific direction; it should be compared
carefully with the 2026 finite-particle convergence results cited under A.

### Revised implementation and research order

The literature-aware priority order is:

| Rank | Objective | Why it remains valuable | First action |
|---|---|---|---|
| 1 | A1/A2 support-restricted equilibria and collapse stability | Clearest open implication for the actual algorithm | Define stability topology; build atomic stability atlas |
| 2 | Explicit quantitative B under overlap/tightness | Goes beyond Lee--Chun's sequential theorem | Choose one quantitative anti-escape class and test exponents |
| 3 | D1 finite-range mass blindness | Cheap, sharp, and practically interpretable | Formalize the separated two-cluster witness |
| 4 | D2/C3 drift-characteristic classification | Broadest genuinely new kernel question | Screen positive non-companion kernels and isolate the real connectivity condition |
| 5 | E local-to-global/far-field observability | High theoretical novelty | Exhaust atomic counterexample searches by regime |
| 6 | G2 certified bandwidth choice | Direct practical value | Combine stability, ESS, and estimator error into one criterion |
| 7 | F infinite-dimensional converse | High upside but requires new mathematics | Find a replacement for the finite-dimensional elliptic trace argument |

Accordingly, major research effort should not go into C1/C2 or H as presently
framed.  They remain useful formalization or benchmarking projects, but current
literature has already occupied most of their informal mathematical territory.

---

## Citation verification and strategic addendum (2026-07-18, post-review)

All seven 2026 arXiv citations above were fetched and verified to exist
(2604.24196, 2603.10592, 2605.10727, 2605.22795, 2605.05118, 2603.12366,
plus the older background papers).  Two facts material to strategy that the
review did not surface:

1. **Priority and proof technique for the converse itself.**  Lee–Chun
   (2604.24196, v1 2026-04-27) informally proved the arbitrary-measure
   identifiability theorem for the whole companion-elliptic class (Gaussian
   + Matérn, ν > 0) about three months before this repository's Lean
   closure (2026-07-18), by an entirely **different method**: they derive a
   continuity equation for a Wronskian field, apply **DiPerna–Lions
   renormalization** (conservation vs. forced exponential growth ⟹ the
   density vanishes), and finish by Fourier uniqueness.  Ours is a global
   **maximum principle** along the backward gradient flow, elementary
   (Grönwall + Picard–Lindelöf + attained max; no renormalization theory),
   plus Bernstein-subordination smoothing injectivity.  Honest positioning
   of this repository's result: *first machine-checked proof, by an
   independent and substantially more elementary route* — not first proof.
   The two developments converged on the same companion-potential/Wronskian
   structure independently (this repository has no public remote).
2. **Why the proof difference matters for Direction B.**  The
   renormalization argument is intrinsically exact: it plays a conservation
   law against exponential growth, and an `ε`-perturbed field breaks the
   conservation identity in a way that is hard to quantify.  The maximum
   principle, by contrast, is assembled from uniformly quantitative
   ingredients (explicit `τ/e` speed, Hessian bound `2`, `ψ_q ≤ τ²`,
   uniform flow time, explicit growth factor `e^{δT}`).  **Our technique,
   not theirs, is the natural vehicle for the explicit-rate stability
   theorem that Lee–Chun leave open** — which upgrades B from "recently
   partially answered" to the flagship theory direction of this repository,
   with a proof asset competitors do not have.

Consequences folded into the working order: the empirical program (A-atlas,
B-exponent hunt, D2/C3 kernel screens, E falsification, G2 design rule,
H benchmark) is unchanged; the theory flagship is B-with-rates via the
perturbed maximum principle; D1 remains the next cheap Lean theorem; C1
(Matérn tower) is reframed as *formal verification of Lee–Chun's class*,
worth doing but not as a novelty claim.
