# The ℝⁿ Laplace converse — research record and roadmap (2026-07-16)

*Fifth research pass.  Target: the **general** paper-faithful theorem — zero
ℓ²-Laplace mean-shift drift identifies **arbitrary** probability measures on
`ℝⁿ`, every finite `n`.  This document records (a) the exact current distance
to that theorem, (b) four new findings of this pass — two falsifications with
explicit counterexample families, one exact Jacobian identity, and one new
structural reduction (the foliation–cancellation route) that yields a
near-complete paper skeleton for the general case, and (c) the staged plan
G1–G5 with effort/risk.  Prior design record: `LaplaceHigherDim.md`
(§4.6–§4.10); this file supersedes its §4.7(D2.c/D2.d) sequencing.*

---

## 0. Target and current distance

**Target theorem (the "general implication for ℝⁿ"):**

```
theorem laplaceZeroDrift_identifies_euclidean
    (τ : ℝ) (hτ : 0 < τ) (n : ℕ) (p q : Measure (EuclideanSpace ℝ (Fin n)))
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) :
    p = q
```

with no support/moment/atom hypotheses — the exact analogue of the closed 1-d
`laplaceZeroDrift_identifies` (LaplaceUnconditionalConverse.lean).

**What is machine-checked today (all axiom-free, `#print axioms` = 3
foundations):**

| asset | statement | where |
|---|---|---|
| 1-d unconditional converse | zero drift ⟹ `p = q`, arbitrary probability measures on ℝ | `LaplaceUnconditionalConverse.lean` |
| Gaussian converse, all dim | same for the Gaussian kernel, any finite-dim inner-product space | `GaussianScoreRecovery.lean:295`, `GaussianArbitraryConverse.lean` |
| L0 potential layer | `∇ψ_ν = D_ν` for every finite measure (no moments); zero drift ⟺ `Z_q·D_p = Z_p·D_q` | `LaplaceRadialFoundations.lean:236/:286` |
| L1 far field | zero drift ⟹ exponential-tilt centroids agree per direction (directional moments only) | `LaplaceRadialFarField.lean:117` |
| L2 dimension reduction | affine-isometric transport of drift; collinear pairs reduce to 1-d and are identified | `LaplaceRadialFarField.lean:261/:300` |
| L3 atom alignment | zero drift ⟹ `q({a})·D_p(a) = p({a})·D_q(a)` at EVERY point, arbitrary probability measures, any finite dim (`Nontrivial E`); mass-ratio + atom-rigidity corollaries | `LaplaceConeExtraction.lean:2051/:2082/:2123` |
| **L4 smoothing injectivity, all dim** | `Z_p = Z_q` (finite measures) ⟹ `p = q` on `EuclideanSpace ℝ ι`, every finite `ι` — via Bernstein subordination, no Bessel closed form | `LaplaceRadialFourier.lean:990` |
| **L5 radial converse, n = 3** | zero drift identifies radial mixtures `radialMixture₃ ν` (profiles on `[0,∞)`, first moments, `RadialSlack₃ τ νp`) | `LaplaceRadialInvariance3.lean`, commit 49706e4 |

So the missing mass between "today" and the target is exactly:

1. **G1** — the radial converse for general `n ≥ 3` (and general center);
2. **G2** — removing the named `RadialSlack₃` hypothesis;
3. **G3** — the `n = 2` radial case (log-divergent `m̃'`);
4. **G4** — from radial to **arbitrary** measures: the general endgame.
   This pass produced a new primary route for it (§2, F-E).
5. **G5** — packaging (rotation-invariance ⟺ shell-mixture equivalence;
   statement surface).

L4 already covers every dimension, so **no injectivity work remains anywhere
in the program** — every remaining gap is about forcing `Z_p = c·Z_q`.

---

## 1. Numerics of this pass (scripts `numerics/rn_screen*.py`, seeds inside)

Run: `uv run --with numpy --with scipy python numerics/rn_screen.py` (and
`rn_screen2/3/4.py`).  τ = 1 WLOG; configs scanned over scales 0.1–20.

- **[EXP1a] Linearized injectivity of the drift map holds robustly in ℝ²/ℝ³**
  (the §4.8 "numerics side-quest (ii)" falsification screen, first run).  At
  `p = q` atomic, the response operator on sum-zero atomic perturbations,
  `T[h](x) = (1/Z_p(x)) Σ_j h_j k(x,y_j)(y_j − x − m_p(x))`, has
  `σ_min ≥ 1.86e-2` (relative `≥ 2.2e-3`) over 400 random configurations per
  dimension, 240 probes each.  **No linear counterexample; the ℝⁿ conjecture
  stands.**  (This extends the 1-d full-rank finding to n = 2, 3.)
- **[EXP1b] Nonlinear residual descent always collapses onto `p`** (up to
  weak-topology near-degeneracy: atom-splitting configurations reach small
  residual with small transport gap, as they must — the residual is weakly
  continuous).  No spurious distinct minimizer found.
- **[EXP2/EXP4] BOTH centroid-monotonicity conjectures are FALSE — see F-B,
  F-C below.**  Random scans: min eig `sym(I + Dm) ≈ −0.16`; directional
  along `m̂ ≈ −0.20`; adversarial optimization drives both to `−O(L/τ)`
  (unbounded), matched by closed-form counterexample families to 3 decimals
  (`rn_screen4.py` output).

---

## 2. New findings (F-A … F-F)

### F-A. Exact Jacobian identity for the mean shift

For atomic (hence, by approximation, all) measures, with `X = y − x`,
`u = X/‖X‖`, and `E_w` the kernel-tilted mean at probe `x`:

```
I + Dm_p(x) = (1/τ) · Cov_w( X , u )         (validated numerically to 3 decimals)
```

In 1-d this is the proven `m' + 1 = (1/τ)Cov_w(X, sgn X) ≥ 0`.  The n-d sign
question is exactly the sign of the *symmetrized* covariance of the
displacement against its own direction field.

### F-B. §4.6(e) centroid monotonicity is FALSE — unboundedly (n ≥ 2)

The conjectured monotonicity of `x ↦ x + m_p(x)` fails, and not marginally:
2-atom family — atom 1 at the probe (offset δ), atom 2 at distance `L` in
direction `u₂` with `⟨u₂,e₁⟩ = c`, masses tuned so the effective kernel
weights are ½ each.  Then

```
Cov_w(X,u) = (L/4)·u₂ ⊗ (u₂ − u₁)  + O(δ),
λ_min sym(I+Dm) = (Lc/4)(c−1)/τ  → −∞   (c = ½: −L/(16τ)).
```

Measured: −0.624/−1.874/−6.249/−18.749 at L = 10/30/100/300 (prediction
−0.625/−1.875/−6.250/−18.750).  The violating direction is transverse to `m`
(along `m̂ = u₂` this family gives `+(L/4)(1−c) > 0`).

### F-C. The directional refinement is ALSO false — unboundedly

The natural repair — require monotonicity only **along the drift direction**,
`1 + ⟨m̂, Dm_p·m̂⟩ ≥ 0` (what a flow-line/characteristic propagation would
consume; equals `m̃'+1 ≥ 0` in the radial/1-d cases) — fails too: 3-atom
family — atom at the probe (weight `p₁`), symmetric far pair at
`L(ε, ±√(1−ε²))` (weights `p₂/2` each, transverse moments cancel, `m̂ = e₁`):

```
1 + ⟨m̂, Dm·m̂⟩ = − p₁p₂·L·ε(1−ε)/τ  → −∞ .
```

Measured = predicted to 3 decimals at L = 10…300 (`rn_screen4.py` [B]).

**Consequence (route-killer):** *no pointwise Jacobian sign control exists in
`n ≥ 2` — in any direction.*  The 1-d proof's core mechanism (`m'+1 ≥ 0`
pointwise) does not generalize pointwise at all.  The radial case survives
because its sign control is **shell-integrated** — the `(n−1)τ` buffer in RSI
comes from the tangential IBP identity (T), an integrated effect that
pointwise configurations can't see.  Every viable general-`n` attack must use
integrated coefficients (RSI-style, along leaves) or a global argument (F-E).
This retires §4.6(e) as stated and the "conditional Chebyshev-association"
program attached to it.

### F-D. The counterexample screens support the conjecture itself

(EXP1a/1b above.)  Together with the 1-d full-rank finding and the closed
radial n = 3 theorem, all evidence says the general theorem is TRUE; the open
problem is mechanism, not truth.

### F-E. THE FOLIATION–CANCELLATION ROUTE (new; primary route for G4)

Everything below is exact algebra from two machine-checked facts —
`∇ψ_ν = D_ν` (L0) and zero drift `⟺ ∇ψ_p = R̃·∇ψ_q` with
`R̃ := Z_p/Z_q > 0` — plus one classical computation.

**(i) One exterior derivative aligns everything.**  `d(dψ_p) = 0` applied to
`dψ_p = R̃·dψ_q` gives `dR̃ ∧ dψ_q = 0` (a.e.; `R̃` is locally Lipschitz, the
ψ's are `C^{1,1}` since `D` is globally Lipschitz).  Hence on `{m ≠ 0}`:
`∇R̃ ∥ ∇ψ_q ∥ ∇ψ_p ∥ m`, so `ψ_p`, `ψ_q`, `R̃` share level hypersurfaces
(leaves ⟂ m), and locally `ψ_p = G(ψ_q)` with `G' = R̃ > 0`.

**(ii) The Matérn PDE (♦) holds in EVERY dimension.**  With
`g(r) = τ(r+τ)e^{−r/τ}`: `Δg = e^{−r/τ}(r/τ − n)` and

```
(♦)     g − τ²Δg = (n+1)τ²·e^{−r/τ}      ⟹      Z_ν = (ψ_ν − τ²Δψ_ν)/((n+1)τ²)
```

pointwise a.e. (`ψ ∈ C^{1,1}` ⟹ `Δψ` a.e.; classical off supports where ψ is
real-analytic).  *(Re-verified this pass; the constant is dimension-uniform
`(n+1)τ²`.)*

**(iii) The cancellation.**  Substitute `ψ_p = G(ψ_q)` into `R̃ = G' = Z_p/Z_q`
using (♦) for both measures.  `Δψ_p = G'Δψ_q + G''|∇ψ_q|²`, and the `Δψ_q`
terms cancel EXACTLY, leaving the **leafwise dichotomy**: for a.e. `x` in the
leaf `{ψ_q = s} ∩ {m ≠ 0}`,

```
τ²·G''(s)·|∇ψ_q(x)|²  =  G(s) − s·G'(s) .
```

The right side depends only on the leaf.  Hence, leaf by leaf, EITHER

- **(non-degenerate branch)** `|∇ψ_q|` is non-constant on the leaf ⟹
  `G''(s) = 0` AND `G(s) = s·G'(s)`.  On an `s`-interval of such leaves,
  `G(s) = c·s`, i.e. **`ψ_p = c·ψ_q` and (by (♦)) `Z_p = c·Z_q` on that whole
  region — the target conclusion, locally, with NO propagation, NO sign
  control, NO ODE**; or
- **(degenerate branch)** `|∇ψ_q| = φ(ψ_q)` is leaf-constant: `ψ_q` is a
  **transnormal function** (reparametrize `h(ψ_q)` to `|∇(h∘ψ_q)| ≡ 1`: a
  signed-distance function; leaves are parallel hypersurfaces).  Decay
  (`ψ_q → 0` at ∞, `ψ_q > 0`) forces the leaves in an unbounded region to be
  **compact** — tubes `∂(K + B_r)` around a compact focal set `K`.  The
  radial case is exactly `K = {point}`.

**(iv) What the degenerate branch still needs ("tube rigidity").**  Caution
recorded from this pass's own derivation-check: `∇Z_q` is NOT parallel to `m`
in general (only the ratio and the potentials are leaf-aligned), so the
degenerate branch is transnormal but not automatically isoparametric — the
Levi-Civita–Segre classification does not apply for free.  Two attacks:

- *Far-field pin.*  For large leaves, `ψ_q(ru) ≈ τ²e^{−r/τ}·L̃_q(u)`
  (L1 machinery), so leaf-constancy forces `r + τ·log L̃_q(u) ≈ const`:
  the tubes' support function is `τ·log` of the directional tilt transform.
  Tubes around `K` have support function `h_K(u) + r`; matching forces
  `τ·log L̃_q(u) → h_K(u)`.  If one can push the far-field expansion one
  order deeper (the L1 monotone-weight machinery is built for this), `K`
  should collapse to a point for probability measures — then the region is
  **radial about a center** and (translated) G1 machinery finishes there.
- *ODE-in-arc-length bypass.*  In the degenerate branch every relevant scalar
  is a function of the leaf parameter; the radial file's Abel system
  generalizes with the leaves' mean curvature `H(s)` replacing `(n−1)/r`.
  If the (T)-identity analogue survives tube-averaging (it holds exactly for
  spheres; the question is stability under the transnormal geometry), the
  whole L5 skeleton runs without classifying the leaves first.

**(v) Gluing (the n-d trichotomy).**  Constants `c` from the non-degenerate
branch are rigid across leaf components: `R̃ = Z_p/Z_q` is globally continuous
and positive, and on `int{m = 0}` BOTH potentials are locally constant, which
by (♦) forces `Z_p, Z_q` locally constant — and a `k`-smoothed probability
measure with `Z` constant on an open set is massively over-determined
(`Z` real-analytic off supp; `Z → 0` at ∞).  The expected structure of the
final proof is exactly the 1-d trichotomy transplanted to the leaf parameter:
non-degenerate regions carry constants, degenerate regions are radial-like
cores, `{m = 0}` interfaces are handled by continuity + rigidity, and the
global constant is then `1` by total mass via the **finite-measure c•q trick**
(F-F).  L3 (atom alignment) fires at atoms of either measure wherever
`m ≠ 0` — the atoms come pre-matched with ratio `R̃`, which should make the
interface bookkeeping strictly easier than 1-d (where atoms were the hard
part).

**(vi) Honest open points of F-E**, in expected order of difficulty:
1. tube rigidity (iv) — the genuinely new mathematics;
2. gluing across `{m = 0}` and branch boundaries (v) — hard bookkeeping,
   1-d gives the playbook;
3. regularity bootstrap — a.e. leafwise identities ⟹ everywhere structure
   (`C^{1,1}` + real-analyticity off supports + continuity of `R̃`; standard
   but must be done carefully);
4. `n = 1` is excluded throughout (leaves are points) — consistent, it's
   already proved by other means.

**Why this route and not the others.**  It strictly sharpens §4.6(d)'s
max-principle reformulation (the potential equation *is* the foliation
statement plus (♦)); it explains structurally why the radial case is the
irreducible core (radial ⟺ everything in the degenerate branch — the
dichotomy is silent there, which is why L5 needed the ODE fight); and it is
consistent with F-B/F-C (it never asks for pointwise sign control — the only
sign-controlled step left is *inside* the radial core, where shell-integrated
RSI is available).  §4.7's other D2.d candidates stay as fallbacks: the
Fourier bilinear form (wall, unchanged), and the odd-`n` local-operator tower
(`(1−τ²Δ)³ψ ∝ ν` in ℝ³ — order `n+3` local; a Lean-friendlier but
longer-range alternative for the gluing step specifically).

### F-F. The mass endgame is dimension-free (carried over from L5)

`c = 1` needs no Tonelli/ray-mass identity in any dimension: apply L4's
finite-measure `LaplaceSmoothingInjective` to `(p, (ENNReal.ofReal c) • q)`
— the normalizer hypothesis is exact via `integral_smul_measure` — and
evaluate at `univ`.  Proved and deployed at n = 3
(`laplaceZeroDrift_identifies_of_radialMixture₃`); the same 20 lines work
verbatim for every `n` and for the final general theorem.

---

## 3. The staged plan

### G1 — radial converse, general `n ≥ 3`, general center  *(next; mechanical-plus)*

Load-bearing twice over: it is the user-visible generalization AND the
degenerate-branch core consumed by F-E(iv).  Generalize the five L5 files
(`Shell3/Ray3/System3/Converse3/Invariance3 → *N` or parametrized versions):

- **Measure/chart design (the one real decision).**  n = 3 used the explicit
  chart `(u,φ) ↦ (u, √(1−u²)cosφ, √(1−u²)sinφ)` with uniform `chartBase`.
  For general `n` the zonal weight is `(1−u²)^{(n−3)/2}`.  Two options:
  - *(a) zonal-only base (recommended):* every ray/shell object in the proof
    is zonal (a function of `(s,u)`), so define `chartBaseN` directly as the
    weighted measure `(1−u²)^{(n−3)/2} du` on `[−1,1]` (normalized) — the
    S-layer (per-shell T-identity, IBP) is genuinely 1-d in `u`, exactly as
    §4.9(R2) derives it: `∂_u(1−u²)^{(n−1)/2} = −(n−1)u(1−u²)^{(n−3)/2}`.
    The *measure* `radialMixtureN` on `ℝⁿ` still needs a genuine sphere
    measure for the statement; build it as `ν.bind (fun s => uniformShell s)`
    with `uniformShell` from Mathlib's polar decomposition
    (`measurePreserving_homeomorphUnitSphereProd` / `Measure.toSphere`) and
    prove ONE bridging lemma: sphere-average of a zonal function =
    `chartBaseN`-average (recon: whether Mathlib has the zonal pushforward;
    if not, it is a self-contained computation worth contributing).
  - *(b) iterated explicit chart:* stay fully explicit with an
    `(n−1)`-variable chart.  Rejected: the θ-flow invariance and all
    boundary-vanishing arguments multiply in complexity.
- **Rotation invariance for free (design improvement over n = 3).**  With
  `uniformShell` *manifestly* isometry-invariant (definition via
  `Measure.toSphere` inherits invariance from `volume`), the master radiality
  theorem `Z_ν(x) = Z̃_ν(‖x‖)` follows from **L2's existing transport lemmas**
  (`kernelNormalizer_laplace_affineIsometryMap` at a rotation mapping `x` to
  `‖x‖·e₁`) — the entire θ-flow/tilt file (1000+ lines at n = 3) collapses to
  a page.  *(The θ-flow was only ever needed because `radialMixture₃` was
  chart-defined without manifest invariance.)*
- **Constant threading.**  `3 ↦ n`, closure `C̃ = Q̃ + nτZ̃ + ((n−1)τ/r)D̃`,
  K̂-coefficient `m̃'+4 ↦ m̃'+(n+1)`, `v = r²w ↦ r^{n−1}w`, `|K̂| ≤ τr^{n−1}`
  edge, `(n−1)/2`-moment hypothesis at ∞, `RadialSlack₃ ↦ RadialSlackN`
  (`−3 ↦ −n`).  The 1-d propagation layer is consumed through the same
  `μ̂ = (m̃'+n+1)/2` wrapper — §4.9(R3)'s dual-slot caveat was already handled
  at n = 3 by FTC-primitive edge cores, which are `n`-generic.
- **General center:** `radialMixtureN ν c := map (c + ·) (radialMixtureN ν)`;
  drift/normalizer transport under translation is an affine isometry — L2
  lemmas apply.  Cheap; do it in the same pass (F-E needs it).
- Estimate: 2–3 sessions.  Risk: low-medium, concentrated in the
  `uniformShell`/zonal-bridge recon.

### G2 — remove `RadialSlack` *(parallel research; makes G1 unconditional)*

The open case is `m̃ > r` (probe inside the centroid).  §4.10's sketch: law of
total covariance over shells — within-shell `Cov_w(X, X/d) ≥ 0` holds by
co-monotonicity on each shell (`s > r` shells verified); the cross-shell term
is where the `(n−1)τ` buffer from (T) must be spent.  F-B/F-C sharpen the
search space: any proof MUST consume shell-integrated structure — do not
spend time on pointwise/association arguments (now provably dead).  Numerics
(RsiScan, 2026-07-15): 40× margin, worst case single-shell just behind the
probe.  If it resists: it stays a named hypothesis and G1 ships conditional
like L5; the F-E route inherits the same condition on its radial cores.
Estimate: open-ended; timebox 1 session per attempt.

### G3 — `n = 2` radial *(after G1)*

Shell-through-probe integrals are log-divergent ⟹ `m̃'` exists only as an
integrable a.e.-derivative at shell radii.  The 1-d file already runs its
propagation in exactly this regime (right-derivatives, FTC form, càdlàg
coefficient layer) — port the G1 skeleton with the 1-d one-sided toolkit.
Also `(1−u²)^{(n−3)/2}` is singular at `u = ±1` for n = 2 (integrable);
the (T) IBP boundary terms still vanish (verified in §4.9(R2) for n = 2).
Estimate: 2–3 sessions.  Not load-bearing for G4 (F-E's degenerate cores in
`ℝ²` are 1-parameter tubes; but general-n statement wants it eventually).

### G4 — the general endgame *(paper-first, start now in parallel)*

Development track for F-E, in order:
- **P1** (regularity): write out the a.e. leafwise dichotomy rigorously for
  `C^{1,1}` potentials; identify exactly where real-analyticity off supports
  is needed.  Deliverable: a self-contained paper section.  1 session.
- **P2** (tube rigidity): the far-field pin — extend L1's monotone-weight
  expansion one order in `1/r` and show leaf-constancy at ∞ forces the focal
  set to a point (probability measures).  This is the single most valuable
  open computation in the program.  1–2 sessions paper + numerics
  (tube-profile fits on synthetic transnormal candidates).
- **P3** (gluing): transplant the 1-d trichotomy playbook to the leaf
  parameter; classify what happens at `int{m=0}` via the `Z`-constancy
  rigidity (F-E(v)); L3 handles atoms at interfaces.  1–2 sessions paper.
- **P4** (assembly + Lean feasibility review): decide the formalization
  boundary — candidates for Paperaxioms allowlisting under the project's
  axiomatize-standard-results policy: Rademacher/Alexandrov-type a.e.
  differentiability (if Mathlib still lacks what's needed) and, if the
  classification route is taken instead of the ODE bypass, the transnormal
  structure theorem.  Target: formalize the skeleton with the radial core
  (G1) plugged in as-is.
- Fallback for P2/P3 if tube rigidity stalls: the odd-`n` local tower
  (`(1−τ²Δ)³ψ ∝ ν` in ℝ³) — pursue `n = 3` general-measure as the first
  unconditional non-radial theorem, accepting dimension-specific scope.

### G5 — packaging *(cheap, with G1/G4)*

- `IsRadial μ ⟺ ∃ ν, μ = radialMixtureN ν` (Haar orbit-average equivalence)
  — optional honesty lemma; the mixture form already IS the honest class.
- Final statement surface + audit entries + `ResearchStatus.md` update.

### Sequencing

```
G1 (radial n≥3 + center)  →  G3 (n=2)          [Lean track]
G2 (slack removal)         — parallel, timeboxed attempts
G4 P1→P2→P3 (paper track)  — start immediately; P2 is the crux
G4 P4 (Lean endgame)       — after G1 + P3
G5                          — opportunistic
```

Total to an unconditional general-ℝⁿ theorem: realistically G1(2–3) + G3(2–3)
+ P1–P3(3–5 paper) + P4(4–8 Lean) sessions, PLUS whatever G2/P2 research
resistance adds.  The two genuine research risks, in order: **P2 tube
rigidity** and **G2 slack removal**; everything else is engineering with
existing templates.

---

## 4. Route obituary (do not re-open without new ideas)

- Pointwise centroid monotonicity, full or directional (§4.6(e)) — **falsified
  this pass**, unboundedly, both forms (F-B/F-C, `rn_screen4.py`).
- Exposed-point/probe-limit atom peeling — no-concentration principle
  (§4.6(b), 2026-07-14).
- Far-field hierarchy alone — constrains only moment ratios (§4.6(b)).
- Pointwise/Frobenius propagation across mean-shift zeros in 1-d — resolved
  globally instead (PASS 4/5 records in the project memory); its n-d analogue
  is subsumed by F-E's global structure.
- Tonelli/ray-mass identities for `c = 1` — obsolete everywhere (F-F).

## 5. Cross-references

- `LaplaceHigherDim.md` §4.6–§4.10 — prior findings; §4.6(e) now carries a
  falsification pointer to this file; §4.10 log has the dated entry.
- `LaplaceL5_HANDOFF.md` — the closed n = 3 milestone (template files for G1).
- `numerics/rn_screen.py, rn_screen2.py, rn_screen3.py, rn_screen4.py` — this
  pass's experiments (seeds 20260716–18; all outputs quoted in §1/§2 verbatim
  from the runs of 2026-07-16).

## 6. G1 implementation status (2026-07-16)

The first two general-`n` layers are now present and compile axiom-free:

* `LaplaceRadialShellN.lean` defines the weighted zonal shell profiles for
  every `n ≥ 3` and proves the tangential identity
  `shellRhoSqOverDistN = ((n-1)τ/r) · shellTN`, including the collision shell
  by continuity in the shell radius.
* `LaplaceRadialRayN.lean` packages those profiles into radial ray integrals.
  It proves the support bookkeeping, the mixed tangential identity,
  `T = D + r Z`, the axial `Q` payload, the per-shell companion closure, and a
  ray-level closure theorem.  The latter carries explicit a.e. shell
  integrability plus radial-payload integrability hypotheses; this is the
  honest boundary before the measure/chart bridge and dominated radial
  differentiation are supplied.
* `LaplaceRadialMeasureN.lean` defines the normalized Haar sphere measure,
  the genuine radius-times-direction radial mixture, centered mixtures, and
  the product-integral bridge.  This closes the measure-construction part of
  G1; it does not yet prove the zonal-coordinate pushforward.

These files are intentionally not imported by the project root yet.  The
remaining G1 work is therefore sharply identified: construct the genuine
uniform-shell/radial-mixture measure in `ℝⁿ`, prove its zonal pushforward bridge
to `zonalWeight`, and then discharge the displayed integrability hypotheses so
that the `Z`, `C`, and `D` ray profiles can enter the derivative/system layer.
No general-`n` converse theorem is claimed by this checkpoint.

### G1 bridge boundary (2026-07-16)

`LaplaceRadialZonalBridgeN.lean` isolates the remaining spherical-slicing
identity as the explicit proposition `ZonalSphereBridge`.  The first sphere
coordinate is proved continuous and measurable, and its Haar pushforward is
proved to be a probability measure.  The rewriting lemma
`integral_uniformSphere_zonal_of_bridge` consumes only an explicit bridge
inhabitant; no axiom or hidden assumption was added.  The general-n converse
remains unpromoted until that classical coordinate-density identity and the
ray-level integrability obligations are proved.

### G1 analytic discharge (2026-07-16)

`LaplaceRadialIntegrabilityN.lean` closes the ray-level integrability
obligations.  It proves both square-over-distance payload bounds by the
uniform estimate `d * exp(-d/tau) <= tau * exp(-1)`, establishes the
measurability and uniform boundedness of the resulting shell profiles, and
derives `radialRayCN_eq_closure_of_probability` for every probability profile
supported on `[0,infinity)`.  Thus no first-moment or additional integrability
hypothesis is needed merely to state the general-n companion closure.

The outstanding G1 work is now geometric and structural: prove the actual
Haar-coordinate `ZonalSphereBridge`, then port the n-dependent derivative,
Abel-system, propagation, and invariance-to-global-normalizer layers.  This is
substantially more than the one bridge lemma; no general-n converse is claimed
before those layers exist.
