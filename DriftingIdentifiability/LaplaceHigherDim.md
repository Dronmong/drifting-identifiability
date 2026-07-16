# The higher-dimensional Laplace converse: research findings and implementation plan

*Research pass, 2026-07-14.  Follow-up to `LaplaceEndgame.md` (whose Frontiers A and C
are closed).  This document records a full exploration of Frontier D — extending the
1-d arbitrary-target Laplace converse `laplaceZeroDrift_identifies` to finite
dimension `n` — and gives a concrete implementation plan.  A second, directed
pass on the ℓ²/radial case (same day, after determining the paper's kernel is
ℓ²) added §4.6 (new findings: monotone compensated weights, the
no-concentration principle, ball-average atom alignment, the
difference-potential maximum principle, the centroid-monotonicity conjecture,
classical-calculus feasibility, subordination injectivity) and the
lemma-level milestone plan §4.8 (L0–L5).  A third pass (same day) deep-dived
the radial milestone L5: §4.9 derives the exact cylindrical formulas, the
tangential IBP identity that eliminates the Laplacian layer, the
`v = r^{n-1}w` substitution making the radial system literally 1-d-shaped,
proves the needed sign condition for `m̃ ≤ 0` and at zeros, isolates the one
remaining open case (RSI for `m̃ > 0`), and rewrites the L5 lemma plan (R8) to
implementation grain.*

---

## 0. Executive summary

> **Paper determination (added 2026-07-14, checked against
> `papers/2602.04770v2.pdf`).**  The referenced paper's Eq. (12) is
> `k(x,y) = exp(−(1/τ)∥x−y∥)` "where τ is a temperature and **∥·∥ is
> ℓ2-distance**" (restated in its Appendix: "the ℓ2 distance computed in
> Eq. (12)"; the training pseudocode uses `cdist`, i.e. Euclidean).  **So the
> paper-faithful n-d target is the ℓ²/radial case — the hard one (§4)** — and
> the repo's norm-generic `laplaceKernel` instantiated at `EuclideanSpace ℝ ι`
> is exactly the paper's kernel (in 1-d the two norms coincide, so the closed
> 1-d theorem is paper-faithful as-is).  The ℓ¹/product result of §2 is *not*
> the paper's kernel for `n ≥ 2`: it stays in the plan as a cheap, genuinely
> new adjacent theorem (sklearn's `laplacian_kernel`) and as the tensorization
> meta-theorem, but the primary Frontier-D objective is now the §4.6 staged ℓ²
> program.

**In dimension `n ≥ 2` the phrase "the Laplace kernel" is ambiguous, and the two
natural readings have completely different difficulty.**  The repo's own
`laplaceKernel` (Paperaxioms.lean:250) is *norm-generic* —
`laplaceKernel τ x y = exp (-(1/τ) * ‖x - y‖)` for any `NormedAddCommGroup E` — so
"the n-d Laplace kernel" is a choice of norm on `ℝⁿ`:

| kernel | formula | a.k.a. | status after this pass |
|---|---|---|---|
| **ℓ¹ / product** | `exp (-‖x-y‖₁/τ) = ∏ᵢ exp (-\|xᵢ-yᵢ\|/τ)` | sklearn's `laplacian_kernel`; tensor-Laplace.  *Not the paper's kernel.* | **Reduction to the 1-d theorem found.  Complete proof on paper; every ingredient already in repo/Mathlib.  Ready to implement.** |
| **ℓ² / radial** | `exp (-‖x-y‖₂/τ)` | Matérn-1/2; exponential kernel.  **The paper's Eq. (12) kernel — the primary target.** | **Research-grade open.**  New structural framework found (Matérn-universal companion PDE, displacement potential, vector Abel system); staged partial results proposed; endgame open even on paper. |

**Finding 1 (the tensorization reduction, §2).**  For the ℓ¹ kernel, zero drift in
coordinate `i` at probe `x` is *exactly* the 1-d Laplace zero-drift condition for a
pair of **exponentially tilted slice measures**.  Because Frontier A's
`laplaceZeroDrift_identifies` is *unconditional* — arbitrary probability measures,
no atomlessness, no moments — it applies to every tilted slice (tilts inherit atoms
and heavy tails, so no conditional 1-d theorem would suffice).  The slices then
force `Z_p = Z_q` globally, and iterated 1-d smoothing injectivity
(`laplaceKernelNormalizer_injective`, LaplaceInjectivity.lean:408) recovers `p = q`
rectangle-by-rectangle.  No new hard analysis is needed; the whole n-d theorem is
Fubini bookkeeping around two existing 1-d theorems.  This is a direct payoff of
Frontier A's unconditional strength.

**Finding 1′ (tensorization meta-theorem, §2.6).**  The reduction never uses
Laplace-specific facts beyond (a) a 1-d unconditional converse and (b) 1-d smoothing
injectivity, per coordinate.  So the right thing to formalize is a **product-kernel
meta-theorem**: any tensor kernel `∏ᵢ kᵢ(xᵢ,yᵢ)` whose 1-d factors each satisfy
(a)+(b) has the unconditional n-d converse.  Instantiations: ℓ¹-Laplace with
**per-coordinate bandwidths** (anisotropic/ARD Laplacian), and even **mixed
Laplace–Gaussian tensor kernels** (both 1-d inputs exist for Gaussian too).  Every
future 1-d kernel win then lifts to n-d for free.

**Finding 2 (the ℓ² framework, §4).**  The radial case has genuinely new structure:

- the **displacement potential**: `D_p = ∇ψ_p` where
  `ψ_p(x) = τ ∫ (‖x-y‖ + τ)·e^{-‖x-y‖/τ} dp(y)` — an elementary, dimension-uniform
  identity (the profile is the Matérn-3/2 kernel), provable by direct
  differentiation exactly like the Gaussian score identity;
- the **Matérn-universal companion PDE**: `(1 - τ²Δ) D_p = (n+1) τ² ∇Z_p`, which at
  `n = 1` is precisely the repo's proven companion/Wronskian identities, and which
  holds *verbatim for every Matérn-ν kernel* (only the constant changes);
- the **vector Abel system**: zero drift forces
  `2(Dm)W + (div W)·m = -(n+1)·W` for the Wronskian field
  `W = Z_q∇Z_p - Z_p∇Z_q`, generalizing the 1-d Abel ODE the endgame was built on;
- a **regularity surprise**: for `n ≥ 2` atoms do *not* produce singular parts in
  `ΔZ_p` (the cone `e^{-r/τ}` has locally integrable Laplacian in `n ≥ 2`), so the
  distributional calculus is *cleaner* than in 1-d — the obstruction is not
  regularity but the absence of the 1-d order/interval structure in the endgame.

The far-field foundation for ℓ² is *already partially formalized*:
`kernelCentroid_laplace_radial_tendsto` (LaplacianGaussianConverse.lean:304) proves,
for arbitrary probability measures with exponential moments, that the kernel
centroid along radial probes converges to the exponential-tilt centroid — in any
dimension.  §4.6 stages the ℓ² program into four milestones of increasing risk.

**Recommendation (revised after the paper determination; concretized by the
directed ℓ² pass).**  The paper-faithful objective is the ℓ² program, now
planned to lemma level as milestones **L0–L5** (§4.8): L0 foundations
(`ψ`, `D = ∇ψ`, potential alignment) → L1/L2 far field + dimensional reduction
→ **L3 atom alignment** (a new provable-now n-d theorem for arbitrary
measures, §4.6(c)) → L4 n-d smoothing injectivity → **L5 radial-measure
converse** (unconditional if the radial-L3 monotonicity lemma falls, else a
named-hypothesis conditional).  Numerics side-quests (§4.8) gate the endgame.
Finding 1 (ℓ¹) remains a cheap, fully-derisked adjacent theorem — worth
implementing at some point for the meta-theorem and the sklearn-kernel
coverage, but it is no longer the headline next step.

---

## 1. Setting, and what "n-dimensional" means here

All drift objects are already dimension-generic in the repo: `meanShift`,
`kernelNormalizer`, `meanShiftDrift`, `ZeroDrift` are defined over any
`[MeasurableSpace E] [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]`
(Paperaxioms.lean:83–213), and the drift integrand is *uniformly bounded*:
`‖k x y • (y - x)‖ ≤ sup_{r≥0} r·e^{-r/τ} = τ/e` for the Laplace kernel of any
norm, so every mean-shift object exists for arbitrary finite measures — no moment
hypotheses anywhere (the file LaplacianGaussianConverse.lean uses a Gaussian first
moment for its `k • y` variant; for `k • (y-x)` even that is unnecessary).

Norm choice.  On `ℝⁿ`:

- `E = EuclideanSpace ℝ ι` gives `laplaceKernel τ x y = exp (-‖x-y‖₂/τ)` — the
  radial/Matérn-1/2 kernel (this is what the existing n-d Gaussian-target file
  LaplacianGaussianConverse.lean works with);
- `E = PiLp 1 (fun _ : ι => ℝ)` gives `exp (-‖x-y‖₁/τ) = ∏ᵢ exp (-|xᵢ-yᵢ|/τ)` —
  the product kernel.  Mathlib provides `WithLp.measurableSpace` and Borel
  instances (Mathlib/Analysis/Normed/Lp/MeasurableSpace.lean), so measures on
  `PiLp 1` are well-formed and the repo's `laplaceKernel`/`meanShiftDrift` apply
  verbatim.

So both n-d questions are literally about the *same* repo definition on different
normed structures.  There is no third canonical choice (ℓ^∞ gives
`exp(-max|xᵢ-yᵢ|/τ)`, which has neither tensor nor radial structure; nothing below
applies to it, and it is not used in practice).

Both conjectures pass the basic sanity screens: the kernels are bounded,
continuous, strictly positive, with everywhere-positive Fourier transforms — so
smoothing is injective and no cheap counterexample mechanism (support escape,
transform zero, collapsed feature) is available.  The Gaussian n-d converse
(`gaussianZeroDrift_identifies`) and the 1-d Laplace converse both being true makes
the ℓ² conjecture very plausible; the ℓ¹ case is *proved* below.

---

## 2. Finding 1: the ℓ¹/product case reduces to the 1-d theorem

Throughout §2: `ι` a `Fintype`, `E = Π i : ι, ℝ` (pi measurable structure; any of
the equivalent norms), `k₁(τ) : ℝ → ℝ → ℝ` the 1-d Laplace kernel, and
`K(x,y) = ∏ᵢ k₁(τᵢ) (x i) (y i)` the product kernel with bandwidth vector
`τ : ι → ℝ`, each `τ i` valid.  `p, q` probability measures on `E`,
`Z_p(x) = ∫ K(x,y) dp(y) > 0` everywhere.

**Hypothesis (n-d zero drift).**  `ZeroDrift (meanShiftDrift K) p q`, i.e. for
every probe `x`:

```
(Z_p x)⁻¹ • ∫ K x y • (y - x) ∂p  =  (Z_q x)⁻¹ • ∫ K x y • (y - x) ∂q .
```

### 2.1 Step 1 — slice transfer: coordinate drift is 1-d drift of a tilted measure

Fix a coordinate `i` and freeze the other probe coordinates `x_{-i}`.  Define the
**tilted slice** of `p`:

```
p̃ := Measure.map (fun y => y i) (p.withDensity (fun y => ∏_{j ≠ i} k₁(τⱼ) (x j) (y j)))
```

a finite measure on `ℝ` (weight `≤ 1`), with strictly positive mass (weight `> 0`,
`p` probability).  Reading off coordinate `i` of the n-d drift (apply the
projection CLM under the Bochner integral) and unfolding `map`/`withDensity`:

```
(meanShift K p x) i = meanShift (k₁ τᵢ) p̃ (x i),
```

because the numerator `∫ K x y · ((y-x) i) ∂p = ∫ k₁(τᵢ)(xᵢ, t)·(t - xᵢ) ∂p̃(t)` and
`Z_p x = ∫ k₁(τᵢ)(xᵢ, t) ∂p̃(t)` — the tilt absorbs exactly the other factors, and
crucially **the tilt does not depend on `x i`**.  Mean shift is invariant under
scaling the measure, so with `P̃ := (p̃ ℝ)⁻¹ • p̃` and `Q̃` likewise:

> n-d zero drift at all `(x i, x_{-i})`, coordinate `i`
> ⟹ `ZeroDrift (meanShiftDrift (k₁ τᵢ)) P̃ Q̃` — an honest 1-d hypothesis for an
> honest pair of *arbitrary* probability measures on `ℝ`.

### 2.2 Step 2 — the 1-d unconditional theorem fires on every slice

`laplaceZeroDrift_identifies (τ i) _ P̃ Q̃` gives `P̃ = Q̃`, i.e.

**(Tᵢ)**  `p̃ = c_i(x_{-i}) · q̃` as measures on `ℝ`, with
`c_i(x_{-i}) = p̃(ℝ)/q̃(ℝ)` the tilt-mass ratio, for **every** `i` and every
`x_{-i}`.

This step is where Frontier A's unconditionality is load-bearing: `p̃` inherits
every atom and every heavy tail of `p`, so the atomless/compact/moment-conditional
1-d theorems could not have powered the reduction.

### 2.3 Step 3 — ratio constancy and `κ = 1`

Integrating the (Tᵢ) measure identity against `t ↦ k₁(τᵢ)(x i, t)` reassembles the
full normalizer:

```
Z_p x = c_i(x_{-i}) · Z_q x    for every x, every i.
```

So `x ↦ Z_p x / Z_q x` is independent of `x i` for *each* `i`; changing one
coordinate at a time, it is a constant `κ` (pure algebra, no continuity needed).

`κ = 1` by peeling coordinates with **iterated 1-d Tonelli** (no n-d product
integral needed): from `p̃ = κ q̃`, total masses give
`∫ ∏_{j∈S} k₁(τⱼ)(x j, y j) ∂p = κ ∫ (same) ∂q` for `S = ι \ {i}` and all centers;
integrating the center `x j` over `ℝ` for one `j ∈ S` replaces the factor
`k₁(τⱼ)` by the constant `2τⱼ` (Tonelli for nonnegative integrands, then
`∫_ℝ e^{-|s|/τ} ds = 2τ`), producing the same identity for `S \ {j}`.  After
`|S|` steps: `1 = p(E) = κ·q(E) = κ`.

Hence **`Z_p = Z_q` everywhere, and every (Tᵢ) is an equality** `p̃ = q̃`.

### 2.4 Step 4 — from slice equalities to `p = q` (the `H(S)` induction)

For a `Finset S ⊆ ι`, say **`H(S)`** holds if for all measurable `A : ι → Set ℝ`
and all centers `x`:

```
∫ ∏_{i∈S} 1_{A i}(y i) · ∏_{j∉S} k₁(τⱼ)(x j, y j) ∂p  =  (same for q).
```

- `H(∅)` is `Z_p = Z_q` (Step 3).
- `H(S) → H(S ∪ {i})`: freeze the `S`-indicators and the centers outside
  `S ∪ {i}`; both sides of `H(S)`, as functions of `x i`, are 1-d Laplace
  smoothings of the finite measures
  `μ_p = map (eval i) (p.withDensity (∏_{S} 1_{A}·∏_{∉S∪i} k₁))` and `μ_q`;
  equal smoothings force `μ_p = μ_q` by
  **`laplaceKernelNormalizer_injective`** (finite measures — exactly its stated
  generality), and evaluating on `A i` gives `H(S ∪ {i})`.
- `Finset.induction` up to `S = univ`: `p` and `q` agree on all measurable boxes
  `{y | ∀ i, y i ∈ A i}`.  Boxes are a π-system generating `MeasurableSpace.pi`
  (`isPiSystem_pi`, `pi_eq_generateFrom` — Mathlib/MeasureTheory/Constructions/Pi.lean),
  so `MeasureTheory.ext_of_generate_finite` closes: **`p = q`**. ∎

The same induction with `H(∅)` taken as a hypothesis is the standalone statement
`productLaplaceSmoothingInjective : Z_p = Z_q → p = q` for finite measures — worth
surfacing separately (it instantiates the repo's own predicate
`LaplaceSmoothingInjective` at `PiLp 1` after the norm bridge).

### 2.5 Why this does not touch the ℓ² kernel

`exp(-‖x-y‖₂/τ)` does not factor through coordinates, so no tilt makes a slice of
the ℓ² drift equal a 1-d drift.  The tensor structure is essential; the reduction
is an ℓ¹ phenomenon.  (Conversely, nothing in §4's PDE framework applies to ℓ¹:
`‖·‖₁` smoothing satisfies a *per-coordinate* identity
`(1 - τ²∂ᵢ²) D_{p,i} = 2τ² ∂ᵢ Z_p`, which is how one re-derives §2 analytically —
a good cross-check, not a needed one.)

### 2.6 The meta-theorem formulation (recommended)

Steps 1–4 use only these per-coordinate facts about the factor kernels
`kᵢ(x,y) = φᵢ(x - y)`:

1. `φᵢ` continuous, strictly positive, `φᵢ ≤ Cᵢ`, `∫_ℝ φᵢ < ∞`,
   `sup_t |t|·φᵢ(t) < ∞`  (definedness + Tonelli constant);
2. **1-d unconditional converse** for `kᵢ`: zero `meanShiftDrift kᵢ` drift between
   arbitrary probability measures on `ℝ` forces equality;
3. **1-d smoothing injectivity** for `kᵢ` on finite measures.

So formalize the **product-kernel tensorization meta-theorem** with (1)–(3) as
hypotheses, then instantiate:

- **ℓ¹-Laplace, per-coordinate bandwidths** `τ : ι → ℝ` (inputs:
  `laplaceZeroDrift_identifies`, `laplaceKernelNormalizer_injective`) — the
  anisotropic/ARD Laplacian kernel; the equal-bandwidth corollary on
  `PiLp 1 (fun _ : ι => ℝ)` is then literally about `laplaceKernel τ` via
  `‖u‖₁ = ∑ |uᵢ|` (`PiLp.norm_eq_sum` at `p = 1`), giving headline parity with the
  1-d statement;
- **mixed Laplace/Gaussian tensor kernels** (inputs for Gaussian factors:
  `gaussianZeroDrift_identifies` at `E = ℝ`, `gaussianKernelNormalizer_injective`)
  — a strictly-new theorem class for free;
- (sanity corollary) pure product-Gaussian — re-proves a case of the Gaussian
  converse by a completely different route.

If hypothesis-plumbing turns out heavier than expected mid-implementation, fall
back to the Laplace-only version — the lemma skeleton is identical.

---

## 3. Implementation plan for Finding 1

New file `ProductKernelConverse.lean` (or split as noted).  Estimated total
600–900 lines.  No new axioms; everything cites existing repo theorems.

**Stage P0 — kernel and slice infrastructure (~150 lines).**
- `productKernel (φ : ι → ℝ → ℝ → ℝ) : (Π i : ι, ℝ) → _ → ℝ := fun x y => ∏ i, φ i (x i) (y i)`;
  positivity, `≤ ∏ Cᵢ`, continuity, measurability; drift-integrand bound `τ/e`-style.
- `sliceTilt p i x : Measure ℝ := (p.withDensity w).map (fun y => y i)` with
  `w y = ∏_{j ≠ i} φ j (x j) (y j)` (as an `ℝ≥0`-density for
  `integral_withDensity_eq_integral_smul`); finiteness, mass positivity
  (`integral_pos`-style, weight strictly positive), and the two rewrite lemmas
  `∫ f dμ̃`-vs-`∫ (f ∘ eval i)·w dp` (`integral_map` + `integral_withDensity_eq_integral_smul`).

**Stage P1 — slice transfer (~120 lines).**
- Coordinate extraction: `(∫ v ∂p) i = ∫ (v · i) ∂p` via
  `ContinuousLinearMap.proj i` and `ContinuousLinearMap.integral_comp_comm`
  (integrand bounded ⟹ integrable).
- `meanShift_productKernel_coord : (meanShift K p x) i = meanShift (φ i) (sliceTilt p i x) (x i)`.
- Scale invariance `meanShift k ((c : ℝ≥0∞) • μ) = meanShift k μ` (via
  `integral_smul_measure`; `c ≠ 0, ∞`).
- `zeroDrift_slice : ZeroDrift (meanShiftDrift K) p q → ∀ i x₋ᵢ, ZeroDrift (meanShiftDrift (φ i)) P̃ Q̃`.

**Stage P2 — slices fire; ratio constancy; κ = 1 (~180 lines).**
- Normalized-tilt probability instances (`(μ univ)⁻¹ • μ`; check for an existing
  Mathlib instance, else a 3-line construction).
- `(Tᵢ)` as a measure identity `p̃ = κᵢ • q̃` (ENNReal-smul bookkeeping from
  `P̃ = Q̃`).
- Ratio constancy (finite chain over coordinates — pure rewriting).
- The Tonelli peel: one lemma
  `tiltMassProp_peel : (S-masses proportional ∀ centers) → ((S\{j})-masses proportional)`
  using `lintegral_lintegral_swap` (nonneg, assumption-free) and
  `∫_ℝ φⱼ(s,t) ds = const` (for Laplace: `= 2τⱼ`, a 20-line half-line exp
  computation if not already present); `Finset.induction` down to `∅`; conclude
  `κ = 1`, `Z_p = Z_q`, and upgraded slice equalities `p̃ = q̃`.

**Stage P3 — `H(S)` induction and headline (~180 lines).**
- `H(S)` as stated in §2.4; step lemma via the same slice-tilt rewrites plus the
  per-factor injectivity hypothesis (Laplace instantiation:
  `laplaceKernelNormalizer_injective`); `Finset.induction`.
- Boxes π-system + `ext_of_generate_finite` (mass equality: both probability).
- Headliners:
  `productKernel_zeroDrift_identifies` (meta-theorem),
  `productLaplaceZeroDrift_identifies` (τ : ι → ℝ version),
  `productLaplaceSmoothingInjective`,
  PiLp-1 bridge `laplaceZeroDrift_identifies_l1` (kernel-equality lemma
  `laplaceKernel τ = productKernel (laplace factors)` on `PiLp 1` via
  `PiLp.norm_eq_sum`, `Real.norm_eq_abs`, `Real.exp_sum`-direction, then transport),
  mixed Laplace–Gaussian corollary, legitimacy surface
  (`BothProbability`-based, mirroring GaussianArbitraryConverse.lean).

**Stage P4 — audit + docs (~30 min).**  AxiomAudit entries for the new
headliners; `lake build --wfail`; `scripts/Check.ps1`; update `ResearchStatus.md`,
`LaplaceEndgame.md` §5, this file's status banner.

**Verify-at-implementation checklist** (all located this pass, names may drift):
`ext_of_generate_finite` (Mathlib/MeasureTheory/Measure/Typeclasses/Finite.lean:448),
`isPiSystem_pi` / `pi_eq_generateFrom` (Constructions/Pi.lean:263, :754 usage
pattern), `integral_withDensity_eq_integral_smul` (Integral/Bochner/
ContinuousLinearMap.lean:250), `ContinuousLinearMap.integral_comp_comm` (same
file), `WithLp.measurableSpace` (Analysis/Normed/Lp/MeasurableSpace.lean:26),
`PiLp.norm_eq_sum` at `p = 1`, `lintegral_lintegral_swap`, normalized-measure
probability instance, `Measure.integral_smul_measure`,
`gaussianKernelNormalizer_injective`'s exact finiteness hypotheses (for the mixed
instantiation only).

**Risks (all low).**  (i) ENNReal↔ℝ plumbing around `withDensity`/normalization —
tedious, not deep.  (ii) `Finset`-indexed products under `simp` — keep products as
explicit `Finset.prod` over `S.compl`-style sets and avoid `Fin n` recursion
entirely (no `piFinSuccAbove` equivs needed anywhere).  (iii) If the meta-theorem
hypothesis bundle fights elaboration, specialize to Laplace-only and keep the
mixed-kernel version as a follow-up.

---

## 4. Finding 2: the ℓ²/radial case — structure, partials, and the open endgame

Notation: `k(u) = e^{-‖u‖₂/τ}` on `ℝⁿ` (`n ≥ 2` throughout this section unless
said otherwise), `Z_μ = k * μ` the normalizer, `D_μ(x) = ∫ (y-x)·k(x-y) dμ(y)` the
mean-shift numerator, `m = D_p/Z_p = D_q/Z_q` the shared mean-shift field under
zero drift, `W := Z_q ∇Z_p - Z_p ∇Z_q` the vector Wronskian.  Zero drift
cross-multiplied: `D_p Z_q = D_q Z_p` (an identity of bounded Lipschitz
vector-fields — products are unproblematic).

### 4.1 The displacement potential (new; formalization-ready)

Direct differentiation of the radial profile `g(r) = τ(r + τ)e^{-r/τ}` gives
`g'(r) = -r e^{-r/τ}`, hence the **exact, dimension-uniform identity**

```
D_p = ∇ψ_p ,   ψ_p(x) := τ ∫ (‖x-y‖ + τ) e^{-‖x-y‖/τ} dp(y) .
```

`ψ_p` is (up to constants) the **Matérn-3/2 smoothing** of `p`.  This is the ℓ²
analogue of the Gaussian score identity `∇ log Z = σ⁻²·meanShift` that powered
Frontier C — except here the potential is *not* `log Z`, which is exactly why the
Gaussian argument doesn't transfer.  Zero drift becomes the **potential alignment
equation**

```
Z_q ∇ψ_p = Z_p ∇ψ_q .            (PA)
```

Lean note: `hasFDerivAt` of `x ↦ g(‖x-y‖)` needs care only at `x = y` (where the
gradient is `0` and `g` has a genuine critical point since `g'(0) = 0` — the
Matérn-3/2 profile is `C¹` through the diagonal, unlike `k` itself!).  So `ψ_p` is
`C¹` with `∇ψ_p = D_p` *everywhere, for arbitrary finite measures* — no atom
caveat.  This makes (PA) fully rigorous with the repo's standard
differentiation-under-the-integral tools (dominated by `sup r e^{-r/τ}`).

### 4.2 The Matérn-universal companion PDE

The radial computation `g - τ²Δg = (n+1)τ² k` (checked symbolically:
`Δg = e^{-r/τ}(r/τ - n)` off the diagonal) integrates to the **companion identity**

```
(1 - τ²Δ) ψ_p = (n+1) τ² Z_p    ⟺    (1 - τ²Δ) D_p = (n+1) τ² ∇Z_p .   (♦)
```

Consistency check at `n = 1`: (♦) is equivalent to the repo's proven pair
`K = τ·m·W` and `K'⁺ = -τ(m'+2)W` (LaplaceWronskian.lean /
LaplaceGeneralConverseCompanionWronskian.lean) — the 1-d companion normalizer is
the specialization of `ψ`.  Also verified against `p = δ₀`, `n = 1`, directly.

Spectrally, `k̂(ω) ∝ (1+τ²|ω|²)^{-(n+1)/2}` and
`∇log k̂ = -(n+1)τ²ω/(1+τ²|ω|²)` — a *rational* multiplier.  The same computation
for any Matérn-ν kernel `k̂ ∝ (1+τ²|ω|²)^{-(ν+n/2)}` gives literally the same PDE
with constant `2ν+n` in place of `n+1`:

> **Every result proved from (♦)+(PA) holds for the whole Matérn family at once.**
> (Gaussian is the `log k̂` -quadratic degenerate limit where the multiplier is
> local — that's precisely why Gaussian was elementary.)

### 4.3 The vector Abel system

Substituting `D = mZ` into (♦) for `p` and `q`, multiplying crosswise and
subtracting (Leibniz expansion of `Δ(mᵢZ)`):

```
2 (Dm) W + (div W) m = -(n+1) W ,        (★)
```

where `(Dm)ᵢⱼ = ∂ⱼmᵢ`, and `div W = Z_qΔZ_p - Z_pΔZ_q`.  At `n = 1` this is
exactly the proven Abel ODE `2m'W + mW' = -2W`.  Contractions worth recording:

- with `m`:  `div(|m|²W) = -(n+1)⟨m, W⟩`  (a divergence identity — integrable
  structure for energy arguments);
- `W = Z_pZ_q·∇u` with `u := log(Z_p/Z_q)`, so `W/(Z_pZ_q)` is **exact**, and `m`
  is a **scaled gradient** (`m = ∇ψ_p/Z_p`) — both 1-forms in (★) carry potential
  structure that the 1-d proof never had to use.  The goal is `∇u ≡ 0`
  (then `Z_p ∝ Z_q`, the constant is `1` by total mass, and injectivity of radial
  Laplace smoothing — nonvanishing `k̂`, same charFun route as
  `laplaceKernelNormalizer_injective` — gives `p = q`).

### 4.4 Regularity: n ≥ 2 is *cleaner* than 1-d

The 1-d proof fought atoms because `Z''` contains `-2p({x})/τ · δ_x`.  In `n ≥ 2`
this does not happen: `Δk ~ -(n-1)/(τr)·e^{-r/τ}` near the diagonal is locally
integrable and carries **no delta** (the defining fractional PDE
`(1-τ²Δ)^{(n+1)/2} k = c δ` only produces the delta at the full fractional power).
Consequences, for *arbitrary* probability measures when `n ≥ 2`:

- `ΔZ_p = τ⁻²(Z_p - k₂ * p)` with `k₂` the (positive, `L¹`) Bessel/Matérn kernel
  one smoothness step down — an a.e.-defined `L¹_loc` *function*;
- `Z_p, D_p ∈ W^{1,∞}` with `ΔZ_p, ΔD_p ∈ L¹_loc`, and the Leibniz expansion
  behind (★) can be justified distributionally by mollifying *one* factor
  (Friedrichs commutator pattern) — no one-sided-derivative machinery needed;
- the price: `∇Z_p` is discontinuous at atoms (directional derivative picks up an
  isotropic `-‖v‖·p({x})/τ` defect), and `ΔZ_p` is unbounded near atoms.

So the *calculus* of (★) is rigorous-izable for arbitrary measures; what is
genuinely missing in `n ≥ 2` is the 1-d **endgame**: the order structure
(`m' + 1 ≥ 0` via the four-mass decomposition, one-sided uniqueness, interval
trichotomy through the zero set of `m`, Grönwall along `ℝ`) has no direct n-d
analogue.  (★) is one first-order system in `n` unknowns with rough coefficients
degenerating on `{m = 0}` — a research-grade uniqueness problem.

### 4.5 What already exists for ℓ² in the repo

- `radial_identity` / `radial_tendsto` (LaplacianGaussianConverse.lean:30/59):
  `r - ‖r•u - y‖ → ⟪u,y⟫` — the far-field geometry, any inner-product dimension.
- `kernelCentroid_laplace_radial_tendsto` (:304): for arbitrary probability `P`
  with `Integrable (exp(‖y‖/τ))` and `Integrable (exp(‖y‖/τ)·‖y‖)`, the kernel
  centroid along `r•u` converges to the **exponential-tilt centroid** — i.e. the
  drift's radial limit recovers `∇Λ` (log-Laplace gradient) on the sphere of
  radius `1/τ` in transform space.  Distribution-generic; only the subsequent
  parameter extraction is Gaussian-specific.
- `meanShift_eq_kernelCentroid_sub` (:318), `laplaceKernel_integrable` (:670),
  `laplaceKernelNormalizer_pos` (:711): generic bookkeeping, reusable.
- The n-d **Gaussian-target** Laplace converse is closed:
  `laplaceGaussianMeanShiftDrift_identifiesAtZero` (:886) with candidate surface
  (:931–949).

### 4.6 Second-pass findings (2026-07-14, directed ℓ² pass)

**(a) Monotone compensated weights — moment-minimal far field.**
`d/dr [r - ‖r•u - y‖] = 1 - ⟨r•u - y, u⟩/‖r•u - y‖ ≥ 0` (Cauchy–Schwarz), so the
compensated weight `w_r(y) = e^{(r-‖r•u-y‖)/τ}` *increases* to `e^{⟨u,y⟩/τ}`.
Hence `e^{r/τ}·Z_p(r•u) ↑ L_p(u) ∈ (0,∞]` by **monotone convergence for every
probability measure** — no moment hypothesis — and since `w_r` is dominated by
its own limit, the centroid limit needs only the *directional* moment
`∫ e^{⟨u,y⟩/τ}(1+‖y‖) dp < ∞`, strictly weaker than the `exp(‖y‖/τ)` dominators
in LaplacianGaussianConverse.lean.  Moreover the probe term cancels exactly:
zero drift along the ray ⟺ `M_p^w/L_p^w = M_q^w/L_q^w` at every *finite* `r`
(no limit needed for the identity itself).

**(b) No-concentration principle (negative finding; corrects the old D2.c
sketch).**  The kernel bandwidth is fixed: as the probe moves anywhere
(including to infinity), the relative weights of two source points tend to
fixed finite ratios (tilt saturation).  No probe limit concentrates the kernel
on a single atom, so exposed-point/asymptotic-separation ideas **cannot**
recover atoms.  Likewise the full far-field `1/r`-hierarchy only re-states
`m_p = m_q` along rays — it constrains moment *ratios*, never the moment
families separately — insufficient alone, for the same reason 1-d needed the
global ODE.  Every viable attack is exact-identity-based.

**(c) Atom alignment via ball averages — a new provable-now n-d theorem.**
For `n ≥ 2` define the cone-extraction functional
`C_a[f] := lim_{ε→0} (f(a) - ⨍_{B(a,ε)} f) · c_n·τ/ε` (solid-ball averages —
no sphere-measure API needed; `c_n` a dimensional constant).  Then:

- `C_a[Z_μ] = μ({a})`: the atom contributes the cone `-ρ/τ`-term of
  `e^{-ρ/τ}`; the atomless remainder is differentiable at `a` (split the
  measure near/far, kernel Lipschitz — standard), and differentiable functions
  have ball-average defect `o(ε)`;
- `C_a[D_μ · v] = 0` for every fixed `v`: the atom term `(a-x)·e^{-ρ/τ}` is odd
  to leading order and ball averages kill odd terms;
- `C_a` has a product rule on this cone-class (`cone × cone` contributes a
  `ρ²`-type term, defect `o(ε)`).

Applying `C_a` to `D_p Z_q - D_q Z_p ≡ 0` componentwise:

```
q({a}) · D_p(a) = p({a}) · D_q(a)      for every a ∈ ℝⁿ  (n ≥ 2, arbitrary p, q).
```

Combined with zero drift at `a`: wherever `m(a) ≠ 0`,
`p({a})·Z_q(a) = q({a})·Z_p(a)` — atoms come in matched pairs with mass ratio
`Z_p/Z_q(a)`; in particular `p({a}) = 0 ⟺ q({a}) = 0` there.  This is the exact
n-d analogue of the 1-d jump identity `Z'₊ - Z'₋ = -2μ({x})/τ` that the 1-d
endgame consumed, and it is provable now (milestone **L3** below).

**(d) The difference-potential equation and a maximum principle.**
With `Φ := ψ_p - ψ_q` and `ζ := Z_p - Z_q`, zero drift gives `∇Φ = m·ζ` and (♦)
gives `Φ - τ²ΔΦ = (n+1)τ²·ζ`.  Eliminating `ζ`:

```
(n+1)τ² · ∇Φ = m · (Φ - τ²ΔΦ),      Φ = g * (p - q) → 0 at ∞ .
```

The whole conjecture ⟺ this single degenerate linear scalar equation forces
`Φ ≡ 0` (then `ζ ≡ 0` and injectivity finishes).  Max-principle fact: at an
interior positive maximum `x₀` of `Φ`, `∇Φ = 0` and `ΔΦ ≤ 0`, so `m(x₀) ≠ 0`
would force `Φ(x₀) = τ²ΔΦ(x₀) ≤ 0` — contradiction.  Hence **positive maxima
(and negative minima) of `Φ` lie on the zero set of `m`** — common critical
points of both potentials — where moreover `ζ(x₀) ≥ Φ(x₀)/((n+1)τ²) > 0` and
`∇²Φ(x₀) = ζ(x₀)·(Dm)(x₀)` (differentiate the equation at `m(x₀) = 0`).  The
1-d trichotomy pivoted on exactly the zero set of `m`; this is its n-d shadow
and the sharpest known reformulation of the endgame.

**(e) The n-d L3 conjecture (centroid monotonicity).**  All 1-d endgame sign
control came from `m' + 1 ≥ 0` — monotonicity of the mean-shift target map.
**Conjecture: `x ↦ x + m_p(x)` is monotone** (`⟨F(x)-F(x'), x-x'⟩ ≥ 0`) for
every probability `p`, every dimension.  Evidence: proved in 1-d (four-mass
decomposition); the single-atom case is the constant map (boundary case); the
Jacobian is `(1/τ)·Cov_w(y, ∇_y‖y-x‖)` — a covariance of `y` against the
monotone field `unit(y-x) = ∇_y‖y-x‖`, whose *conditional* version on
cylinders around the probe axis is nonneg by Chebyshev association (given
`‖y_⊥‖`, both entries are monotone in `⟨y, e_r⟩`), but the cross-slice term is
uncontrolled — genuinely open; naive association arguments do not close it.
Its radial specialization (`(r + m̃(r))' ≥ 0`) is the key missing lemma of
milestone **L5**.  Run numerics first (random small atomic configurations in
ℝ², search for monotonicity violations); if false, L5 falls back to
double-shooting (see (f)).

**(f) Radial measures make everything classical (Lean feasibility).**  Mathlib
has no usable weak-derivative calculus, so every milestone below is designed
around classical derivatives and integral functionals only — the same
discipline that carried Frontier A.  Decisive observations: a rotation-invariant
measure can only have an atom at the origin, and for such measures
`∫ dμ(y)/‖x-y‖ < ∞` locally uniformly off the origin (a `1/r` singularity
against sphere-uniform mass is integrable for `n ≥ 2`), so `Z_μ, ψ_μ` are
classically `C²` on `ℝⁿ∖{0}` with dominated second derivatives — (♦), (★), and
the radial reduction all become theorems about classical derivatives.  Bonus
boundary data: `W = w(r)·e_r` has `w(0⁺) = 0` (C¹ radial fields are critical at
the origin) *and* `w(∞) = 0`, so the radial ODE can be shot from **both** ends —
the 1-d proof only ever had decay ends.  This is why the radial case is the
right first ℓ² theorem.

**(g) n-d smoothing injectivity without the closed-form transform.**  The final
step everywhere (`Z_p = Z_q ⟹ p = q` on `EuclideanSpace`) does not need the
Bessel closed form of `𝓕(e^{-‖·‖/τ})`.  Complete monotonicity of `e^{-√s}`
gives the subordination `e^{-r} = ∫₀^∞ e^{-r²u}·(4πu³)^{-1/2}e^{-1/(4u)} du`,
so `𝓕(e^{-‖·‖/τ})(t) = ∫₀^∞ (Gaussian transform > 0)·(positive weight) du > 0`
by Tonelli plus Mathlib's Gaussian Fourier integral
(Mathlib/Analysis/SpecialFunctions/Gaussian/FourierTransform.lean).  Then
mirror the proven pattern: Fourier-transform the normalizer and factor
`charFun_p × (profile transform)` by Fubini/translation invariance
(GaussianConvolutionInjectivity.lean:218–288 is the exact template, explicitly
designed to avoid any density-of-convolution API), cancel the nowhere-zero
factor, finish with `Measure.ext_of_charFun`.  Integrability of the profile on
ℝⁿ needs no polar coordinates: `e^{-‖x‖₂/τ} ≤ ∏ᵢ e^{-|xᵢ|/(√n·τ)}`.

### 4.7 Staged program for ℓ² (increasing risk)

**D2.a — far-field foundation, arbitrary targets (low risk, ~1 session).**
From zero drift + exponential moments on both measures:
`exponentialTiltCentroid p τ u = exponentialTiltCentroid q τ u` for all unit `u`
(subtract the existing centroid limits; the probe-`x` term cancels).  Package as:
**zero drift ⟹ ∇Λ_p = ∇Λ_q on the sphere `‖s‖ = 1/τ`** (`Λ` = cumulant
generating function).  This is the n-d analogue of the 1-d "radial-limit
foundation" (parts A–E) and is *known to be insufficient alone* (an analytic
`h = Λ_p - Λ_q` can have `∇h = 0` on a sphere, e.g. `h = (|s|²-1/τ²)²`-shaped;
the convexity of both `Λ`'s is extra structure not yet exploited).  Value:
immediately closes **compactly-supported-or-exponential-moment finite mixture
recovery questions** when combined with D2.c, and is shared infrastructure.

**D2.b — radial (rotation-invariant) measures (medium risk, 1–2 sessions).**
For rotation-invariant `p, q`: `m = m_r(r)·x/r`, `W = w(r)·x/r`, and (★) reduces
to the scalar Abel-type ODE

```
2 m_r' w + m_r (w' + (n-1) w / r) = -(n+1) w    on (0,∞),
```

which is the repo's 1-d Abel equation with a geometric `(n-1)/r` term.  The whole
1-d endgame toolkit (integrating factors, Grönwall propagation, trichotomy over
zeros of `m_r`, edge linear bounds) ports with `r`-weights; `ψ`/(♦) provide the
identities, and by §4.6(f) everything is *classical* for radial measures.
Sign control needs the radial L3 lemma `(r + m̃(r))' ≥ 0` — currently open, see
§4.6(e); fallback: double-shooting from the two vanishing ends `w(0⁺) = 0`,
`w(∞) = 0`.  Finish: `w ≡ 0 ⟹ Z_p ∝ Z_q ⟹ p = q` by radial smoothing
injectivity (§4.6(g)).  This would be the **first genuinely-n-d ℓ²-Laplace
identifiability theorem** and directly stress-tests (♦)/(★) formalization.
Full lemma-level plan: milestone **L5** in §4.8.

**D2.c — finitely supported measures (medium-high risk; route corrected by
§4.6(b)–(c)).**  ~~Exposed-point far-field peeling~~ — invalidated by the
no-concentration principle: no probe limit isolates an atom.  The corrected
route is **singularity matching**: by §4.6(c) (milestone L3), zero drift
already forces the atom sets to coincide with mass ratio `Z_p/Z_q` wherever
`m ≠ 0`.  For finite atomic measures, `Z, D` are analytic off the (finite)
support set, so the identity `D_pZ_q = D_qZ_p` splits into: the L3 relations at
each atom, plus a real-analytic identity on the connected complement (`n ≥ 2`
keeps it connected — this is where n-d is *better* than 1-d).  Remaining work:
show these finitely many relations plus analytic continuation force the
mass-ratio function to be constant `= 1`.  Also the right setting for the
numerical falsification pass (optimize drift residual over small atomic
configurations) *before* investing in the general endgame — recommended,
external to Lean.

**D2.d — general endgame (open research).**
The sharpest known reformulation is now the difference-potential equation of
§4.6(d): `(n+1)τ²∇Φ = m(Φ - τ²ΔΦ)` with decaying `Φ`, whose extrema are pinned
to `{m = 0}` by the maximum principle — the endgame is exactly "rule out a
nontrivial `Φ` peaking on the zero set of `m`", and the n-d L3 conjecture
(§4.6(e)) is the missing sign control.  Other candidate attacks, in order of
current promise: (i) energy/commutator method on (★) using the divergence
identity and `W = Z_pZ_q∇u` (integrate against cutoffs; `W → 0` at infinity
with far-field decay from D2.a machinery); (ii) unique-continuation for `Φ` on
`{m ≠ 0}` using (♦)-ellipticity; (iii) the Fourier bilinear form
(`∫ K̂(ω-ξ)K̂(ξ) F(ω-ξ,ξ) dξ = 0` with `F` the antisymmetrized `p̂⊗q̂`) — the
wall every soft approach hits, likely needs a genuinely new idea; (iv) the
`n = 3` special case first, where `(1-τ²Δ)²` is a *local* 4th-order operator
(odd dimensions are local; even are fractional), so the 1-d "tower reaches the
measure" argument has a literal analogue two derivatives deeper.  Treat as
paper-first; do not open a Lean file for D2.d until L5 and the numerics inform
the structure.

**Dimensional-reduction lemma (free, do with D2.a).**  If `supp p ∪ supp q` lies
in an affine subspace `A`, probes restricted to `A` see exactly the
lower-dimensional ℓ²-radial drift of the restricted measures (ℓ² distances within
`A` are ambient).  So the ℓ² conjecture only needs proving for measures whose
joint support affinely spans; in particular the collinear case is *already closed*
by Frontier A.  Cheap, and it makes D2.b/D2.c statements sharper.

### 4.8 Concrete implementation plan: milestones L0–L5

All milestones use `E` = real inner-product space with the usual instance
bundle (mirror LaplacianGaussianConverse.lean's variable block), specializing
to `[FiniteDimensional ℝ E]` and `n = finrank ℝ E ≥ 2` where dimension enters.
Classical derivatives only (§4.6(f)).  Estimated totals: L0–L4 ≈ 1600–2400
lines over ~5 sessions; L5 ≈ 1000–1800 lines gated on one open lemma.

| # | file | deliverable | effort | risk |
|---|---|---|---|---|
| **L0 ✅ DONE** | `LaplaceRadialFoundations.lean` | `ψ`, `D = ∇ψ`, (PA), `Φ` basics | 1 session | low |
| **L1 ✅ DONE** | `LaplaceRadialFarField.lean` | moment-minimal drift radial limits (D2.a) | 1 session | low |
| **L2 ✅ DONE** | `LaplaceRadialFarField.lean` | affine-isometry dimensional reduction + parametrized collinear corollary | 0.5 session | low |
| L3 | `LaplaceAtomAlignment.lean` | `C_a` functional; atom-alignment theorem | 1–2 sessions | medium |
| L4 | `LaplaceInjectivityEuclidean.lean` | n-d smoothing injectivity | 1–2 sessions | medium |
| L5 | `LaplaceRadialMeasureConverse.lean` | radial-measure converse (D2.b) | 3–5 sessions | high (one open lemma) |

> **✅ L0, L1, and L2 implemented and machine-checked (2026-07-14), axiom-free,
> `--wfail`-clean, wired into the root module.**  Both went through essentially
> as designed.  L0 delivered `matern32Profile`(+ derivative/bounds/`τ·e⁻¹`
> gradient bound), the through-the-diagonal gradient
> `hasFDerivAt_matern32Profile_norm_sub` (chain rule through `√⟪·,·⟫` off the
> diagonal; a `g'(0)=0` little-o argument at `x=y`), the field/potential defs
> with `τ/e`-bounded integrability, the headline
> **`hasFDerivAt_laplaceDisplacementPotential`** (`∇ψ = D` via
> `hasFDerivAt_integral_of_dominated_of_fderiv_le` with the *uniform* `τ/e`
> gradient dominator — no moments), and **`zeroDrift_displacementAligned`**
> (`Z_q·D_p = Z_p·D_q`, proved purely from `meanShift = Z⁻¹·D` + positivity, so
> it does *not* depend on the FDeriv).  L1 delivered
> `laplaceCompensatedWeight_monotone` and the headline
> **`zeroDrift_tiltedCentroid_eq`** (radial-limit reduction through the existing
> `kernelCentroid_laplace_radial_tendsto`).  L2 delivered the reusable transport
> lemmas `kernelNormalizer_laplace_affineIsometryMap`,
> `laplaceDisplacementField_affineIsometryMap`,
> `meanShift_laplace_affineIsometryMap`, the WLOG reduction
> `zeroDrift_of_affineIsometryMap_zeroDrift`, and the one-dimensional
> pushforward corollary `laplaceZeroDrift_identifies_of_collinear`.  The
> collinear statement is intentionally formulated for laws already presented as
> pushforwards along a common line; extracting such a parametrization from an
> abstract support-in-line hypothesis is support/disintegration packaging, not
> part of the analytic L2 core.  Lean notes for the next agent:
> `innerSL` application is `innerSL_apply_apply` (not `innerSL_apply`); the CLM
> norm isometry is `innerSL_apply_norm`; CLM-valued `Continuous.aestronglyMeasurable`
> needs an explicit `haveI : SecondCountableTopologyEither E (E →L[ℝ] ℝ) :=
> ⟨Or.inl inferInstance⟩`; `omit [..] in` must sit **before** the docstring; and
> `2 • innerSL` is cleared with `two_smul` + `add_apply` rather than a smul-type
> guess.  Effort: ~1 session for both (as estimated).

**L0 — foundations.**
- `matern32Profile τ r := τ * (r + τ) * exp (-r/τ)`; elementary lemmas:
  `matern32Profile_deriv : HasDerivAt (matern32Profile τ) (-r * exp (-r/τ)) r`,
  bounds `≤ τ²`-scale, monotone decreasing, `→ 0`.
- `hasFDerivAt_matern32_norm : HasFDerivAt (fun x => matern32Profile τ ‖x-y‖)
  (⟪-(x-y)·e^{-‖x-y‖/τ}, ·⟫) x` — at `x ≠ y` by chain rule through
  `‖·‖ = sqrt ⟪·,·⟫`; at `x = y` directly:
  `|g(‖h‖) - g(0)| ≤ ‖h‖²/2` from `|g'| ≤ r`, giving `IsLittleO` with zero
  derivative (matches the formula, which vanishes at `x = y`).
- `laplaceDisplacementPotential τ μ x := ∫ y, matern32Profile τ ‖x-y‖ ∂μ` (=`ψ`);
  continuity, `0 < ψ ≤ τ(τ + sup r e^{-r/τ}·…)`-bound, `ψ → 0` at `∞` (DCT).
- **`hasFDerivAt_laplaceDisplacementPotential`**: `∇ψ_μ = D_μ` for every finite
  `μ` — via `hasFDerivAt_integral_of_dominated_loc_of_lip`
  (Mathlib/Analysis/Calculus/ParametricIntegral.lean:165) with the *global*
  Lipschitz dominator `τ/e` (`sup_r r e^{-r/τ}`); no moments, no atom caveat.
- `laplaceMeanShiftNumerator_eq_grad_potential` (bridge to the repo's `meanShift`
  via `meanShift_eq_kernelCentroid_sub`-style rewrites; note the drift integrand
  `k • (y-x)` is bounded by `τ/e`, so all integrability is free).
- Potential alignment: `zeroDrift_iff_potentialAligned :
  ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q ↔ ∀ x,
  Z_q x • ∇ψ_p x = Z_p x • ∇ψ_q x` (uses `laplaceKernelNormalizer_pos`).
- `Φ := ψ_p - ψ_q`, `ζ := Z_p - Z_q`; `∇Φ = m·ζ` under zero drift; `Φ → 0`.
  (The (♦)/max-principle layer of §4.6(d) is *deferred* to L5/D2.d — it needs
  `ΔΦ`, i.e. the C² theory; L0 stays first-order.)

**L1 — far field (D2.a), strengthening the existing file.**
- `compensatedWeight_monotone : Monotone (fun r => laplaceCompensatedWeight τ r u y)`
  (derivative sign or direct convexity inequality
  `r ≤ r' ⟹ r - ‖r•u - y‖ ≤ r' - ‖r'•u - y‖`, a triangle-inequality computation).
- MCT normalizer limit for arbitrary probability `P`:
  `Tendsto (fun r => exp (r/τ) * Z_P (r•u)) atTop (𝓝[≤∞] L_P u)` in `ℝ≥0∞`-form
  (`lintegral` + `lintegral_iSup` of the monotone family).
- Directional-moment DCT centroid limit: hypotheses
  `Integrable (fun y => exp (⟪u,y⟫/τ)) P` and `(·)*‖y‖` version; dominator = the
  monotone limit itself (`w_r ≤ w_∞`).  Reuses
  `integral_laplaceCompensatedWeight(_smul)_tendsto` proofs with the smaller
  dominator.
- **`zeroDrift_tiltedCentroid_eq`**: zero drift + the two directional moments at
  `u` for both measures ⟹
  `exponentialTiltCentroid p τ u = exponentialTiltCentroid q τ u`.
  Proof: `m_p(r•u) = m_q(r•u)`, add `r•u`, pass to the limit with
  `kernelCentroid_laplace_radial_tendsto` (existing) generalized to the weaker
  dominators.  Sphere-wide corollary under `Integrable (exp (‖y‖/τ))`.
- Optional cgf phrasing via `Mathlib.Probability.Moments.Tilted` (the Gaussian
  file already exercises this API).

**L2 — dimensional reduction (same file).  ✅ DONE for affine-isometric pushforwards.**
- For an affine subspace `A ⊆ E` (model: `(x₀ +ᵥ S)` with `S` a subspace, or a
  `LinearIsometryEquiv`-embedded `F`), if `p, q` are supported in `A`, then for
  probes `x ∈ A` the drift lies in the direction space of `A` and equals the
  drift of the restricted measures computed inside `A` (distances agree;
  `Measure.map` of the isometry transports the integrals).
- Implemented core: `affineIsometryEmbedding`, distance/kernel preservation,
  normalizer transport, displacement-field transport, mean-shift transport, and
  `zeroDrift_of_affineIsometryMap_zeroDrift`.
- Implemented corollary:
  `laplaceZeroDrift_identifies_of_collinear`: if `p` and `q` are probability
  measures on `ℝ` and their ambient laws are `map (fun t => a + t • u)` for a
  common unit direction `u`, then ambient zero drift identifies the two ambient
  pushforwards, by transport to `ℝ` + `laplaceZeroDrift_identifies` (Frontier A).
  First unconditional n-d ℓ² statement, in parametrized collinear form.
- Remaining packaging, if desired later: convert an abstract hypothesis
  “`P` and `Q` are supported in a common affine line/subspace” into explicit
  source measures whose pushforwards are `P` and `Q`; this is not needed for the
  analytic reduction and was deliberately not folded into L2.
- General WLOG lemma, in implemented form: if the pair is already given as an
  affine-isometric pushforward from `F`, it suffices to prove the ℓ² converse in
  `F`.  A support-spanning formulation can be added on top of the packaging
  lemma above.

**L3 — atom alignment (new theorem).**
- Ball-average operator: `ballAvg f a ε := ⨍ x in Metric.ball a ε, f x`
  (`MeasureTheory.average`; `measure_ball_pos`, `measure_ball_lt_top`).
- Cone extraction on the model function:
  `ballAvg (fun x => exp (-‖x-a‖/τ)) a ε = 1 - κ_n·ε/τ + O(ε²)` with
  `κ_n = n/(n+1)` (compute `⨍_{B(0,ε)} ‖x‖ = n·ε/(n+1)` via the repo-friendly
  route: `∫_{B} ‖x‖` by Fubini on the layer-cake `∫₀^ε volume(B∖B_s)-shells` —
  i.e. `∫_B ‖x‖ dx = ∫₀^ε (volume(B_ε) - volume(B_s)) ds` — needs only
  `EuclideanSpace.volume_ball ∝ r^n`, no surface measure).
- `differentiableAt_kernelNormalizer_of_noAtom`: `μ({a}) = 0 ⟹` `Z_μ`
  differentiable at `a` (split `μ` near/far by `μ(B(a,δ)) → 0`
  (`tendsto_measure_iUnion`-style continuity), Lipschitz kernel on the near
  part, dominated C¹ on the far part).  Corollary: differentiable ⟹ ball-average
  defect `o(ε)` (linear part averages out by symmetry of the ball).
- `ballAvg_coneCoeff_normalizer : Tendsto (fun ε => (Z_μ a - ballAvg Z_μ a ε) * ((n+1)·τ)/(n·ε)) (𝓝[>] 0) (𝓝 (μ {a}))` — the atom/rest split.
- `ballAvg_coneCoeff_numerator_zero`: same limit for each component of `D_μ` is
  `0` (odd-term cancellation `⨍_B (x-a)·φ(‖x-a‖) = 0` by the ball's symmetry —
  provable by the negation map `MeasurePreserving` on the ball).
- Product rule: for `f, g` of the form `cone + differentiable-at-a`, the limit
  functional of `fg` is `f(a)·C_a[g] + g(a)·C_a[f]`.
- **`laplaceZeroDrift_atomAlignment`**: zero drift ⟹
  `∀ a, q {a} • D_p a = p {a} • D_q a` (ℝ≥0∞→ℝ mass coercions inside);
  corollaries `laplaceZeroDrift_atom_iff` (`m(a) ≠ 0 ⟹ (p {a} = 0 ↔ q {a} = 0)`)
  and `laplaceZeroDrift_atomMassRatio` (`p {a} * Z_q a = q {a} * Z_p a` there).
- Hypothesis `2 ≤ n` enters only through the `o(ε)`-defect of the rest (in
  `n = 1` the rest is *not* differentiable-with-o(ε)-average — consistent with
  the 1-d jump identity carrying a different constant).

> **L3 implementation status (2026-07-14).**  Codex implemented `L2` in full and
> the **algebraic gate** of `L3` in `LaplaceAtomAlignment.lean`: given the
> cone-extraction data `LaplaceAtomConeProductData` (the two `Tendsto` facts
> that the ball-average cone coefficient of `Z_ν·D_μ` converges to
> `ν({a})·D_μ(a)`), zero drift forces atom alignment
> (`laplaceZeroDrift_atomAlignment_of_coneProductData`) plus the mass-ratio and
> `p{a}=0 ↔ q{a}=0` corollaries.  What remains for "full L3" is to **discharge
> that hypothesis** — the ball-average asymptotics — which is a large analytic
> build (see the `w(ε)` route below).
>
> **Foundation implemented (`LaplaceConeExtraction.lean`, machine-checked,
> axiom-free).**  Steps (1)–(4) of the route below are done:
> - (1) reflection symmetry: `x ↦ 2a − x` is measure-preserving + a measurable
>   embedding (`measurePreserving_reflection`, `measurableEmbedding_reflection`,
>   `image_reflection_ball`), giving `∫_{B(a,ε)}(x−a) = 0`
>   (`setIntegral_sub_center_ball_eq_zero`) and `⨍_{B(a,ε)}(x−a) = 0`
>   (`setAverage_sub_center_ball_eq_zero`).  Instances:
>   `[(volume).IsAddHaarMeasure]`, `[(volume).IsNegInvariant]` (hold on
>   finite-dim spaces).
> - (2) ball-average defect `o(ε)`: `tendsto_setAverage_defect_of_differentiableAt`
>   — `ε⁻¹•(φ a − ⨍_{B(a,ε)}φ) → 0` for `φ` continuous, differentiable at `a`.
> - (3) `w` lower bound: `kernelAverageDefect τ a ε = 1 − ⨍ e^{−‖·−a‖/τ}` with
>   `kernelAverageDefect_ge : ε/(4eτ) ≤ w(a,ε)` for `0 < ε ≤ 2τ` (annulus
>   fraction `1 − 2^{−n} ≥ 1/2` via `addHaar_ball`; `1 − e^{−s} ≥ s·e^{−1}`).
> - (4) single-kernel facts: `laplaceKernelPt`, its continuity/positivity/`≤1`,
>   and the `1/τ`-Lipschitz bound `abs_laplaceKernelPt_sub_le`
>   (from a from-scratch exp-Lipschitz-on-`Iic 0`, `abs_exp_sub_exp_le_of_nonpos`).
>
> Step (5) is now substantially discharged in `LaplaceConeExtraction.lean`: the
> `w`-normalized single-kernel coefficient has the exact self limit, the
> off-source zero limit, a uniform `4e` domination bound, measurability in the
> source variable, and the DCT integral-form theorem
> `tendsto_integral_kernelAverageConeCoeffW_laplaceKernelPt`, which extracts
> `μ({a})` from `∫ C^w_a[k_y] dμ(y)`.  The Fubini bridge is also proved:
> `tendsto_kernelAverageConeCoeffW_kernelNormalizer_laplace` says the *actual*
> `w`-normalized ball-average defect of `Z_μ` tends to `μ({a})`.  On the
> displacement side, the pointwise fixed-source vector coefficient is proved to
> tend to `0` for every source (`tendsto_kernelAverageConeCoeffWVec_laplaceDisplacementKernel`):
> the self-source case is exact odd symmetry and the off-source case is
> differentiability plus `w ≥ cε`.  The integrated displacement-numerator
> extraction is now also closed:
> `tendsto_kernelAverageConeCoeffWVec_laplaceDisplacementField` proves that the
> actual `w`-normalized ball-average defect of `D_μ` tends to `0`, using a
> vector DCT theorem and a Fubini bridge back to `laplaceDisplacementField`.
> The product rule is closed as well:
> `tendsto_kernelAverageConeCoeffWVec_laplaceNormalizerDisplacementProduct`
> proves that the actual `w`-normalized cone coefficient of `Z_ν · D_μ`
> converges to `ν({a}) · D_μ(a)`, using explicit Lipschitz bounds for `Z_ν`
> and `D_μ` plus a quadratic cross-term estimate.  The final `w`-normalized
> discharge is closed:
> `laplaceZeroDrift_atomAlignment_of_coneExtraction` proves atom alignment
> directly from zero drift, and
> `laplaceZeroDrift_atomMassRatio_of_coneExtraction` /
> `laplaceZeroDrift_atomMass_zero_iff_of_coneExtraction` provide the downstream
> ratio and atom-rigidity corollaries.  The older fixed-scale
> `LaplaceAtomConeProductData` interface remains as a legacy gate; the certified
> L3 route now bypasses it rather than instantiating its hard-coded
> `((n+1)τ)/(nε)` scale.
>
> **The `w(ε)`-normalizer route (recommended; avoids the exact `⨍‖x‖`
> constant).**  Codex's `laplaceAtomConeCoeff` hard-codes the scale
> `(n+1)τ/(nε)`, which forces the exact `⨍_{B(a,ε)}‖x−a‖ = nε/(n+1)` layer-cake
> computation.  Instead normalize by the kernel's own average defect
> `w(a,ε) := 1 − ⨍_{B(a,ε)} e^{−‖x−a‖/τ}` (independent of the measure).  Then for
> `2 ≤ n = finrank ℝ E`:
> 1. **Reflection symmetry** `⨍_{B(a,ε)}(x−a) = 0` — via
>    `Measure.measurePreserving_neg` + translation invariance
>    (`MeasurePreserving.setIntegral_image_emb`); the reflection `x ↦ 2a−x`
>    preserves the ball and negates `x−a`.
> 2. **Ball-average defect of a differentiable function is `o(ε)`**: if `φ` is
>    `DifferentiableAt ℝ φ a` then
>    `φ(a) − ⨍_{B(a,ε)} φ = o(ε)` (Taylor `φ(x)−φ(a)−∇φ(a)(x−a) = o(‖x−a‖)`, the
>    linear part averages to `0` by (1), the remainder is `≤ η(ε)·ε`).
> 3. **`w` lower bound** `w(a,ε) ≥ c·ε` for small `ε` — *no* exact constant, just
>    the annulus ratio `vol(B(a,ε)∖B(a,ε/2))/vol(B(a,ε)) = 1−2^{−n} ≥ 1/2`
>    (`addHaar_ball` + `measure_diff`), combined with `1−e^{−r/τ} ≥ c` for
>    `r ≥ ε/2`.  Also `0 < w`, `w → 0`.
> 4. **Single-kernel differentiability**: for `y ≠ a`, `x ↦ e^{−‖x−y‖/τ}` is
>    `DifferentiableAt ℝ · a` (chain rule, `‖·−y‖` smooth off `y`), so its cone
>    defect is `o(ε)` by (2); for `y = a` the defect is *exactly* `w(a,ε)`.
> 5. **Atom extraction via Fubini + DCT**:
>    `(Z_μ(a) − ⨍_{B(a,ε)}Z_μ)/w = ∫ [ (e^{−‖a−y‖/τ} − ⨍e^{−‖x−y‖/τ})/w ] dμ(y)`
>    (Fubini, `integral_integral_swap`), integrand `→ 1_{y=a}` pointwise (by (3)+(4);
>    at `y=a` it is exactly `1`) and bounded by `1/(cτ)` (Lipschitz + (3)), so DCT
>    gives `→ μ({a})`.  The same machinery gives `(D_μ(a) − ⨍D_μ)/w → 0` (the
>    `y=a` term vanishes by the radial-odd symmetry (1); all others `o(ε)`).
> 6. **Product rule**: `C^w_a[Z_ν·D_μ] = ν({a})·D_μ(a)` from
>    `T_ε[fg] = f(a)T_ε[g] + g(a)T_ε[f] − ⨍((f−f(a))(g−g(a)))`, with the cross
>    term `≤ L_f·ε·ω_g(ε)` giving `o(w)` (f Lipschitz, g continuous at a).
> 7. **Discharge**: combine with `Z_q·D_p = Z_p·D_q` (zero drift,
>    `zeroDrift_displacementAligned`, L0) — the `w`-normalized coefficients are
>    equal for every `ε`, so their common limit gives
>    `q({a})·D_p(a) = p({a})·D_q(a)` **unconditionally**.  Implemented as
>    `laplaceZeroDrift_atomAlignment_of_coneExtraction`; the fixed-scale
>    `LaplaceAtomConeProductData` gate is now a legacy formulation, not the
>    active discharge route.
> All ingredients verified present in Mathlib (`addHaar_ball`, `measure_diff`,
> `lintegral`/`integral_eq_integral_meas_lt` layer cake — used only for (3)'s
> bound, `Measure.measurePreserving_neg`, `integral_integral_swap`,
> `tendsto_integral_filter_of_dominated_convergence`,
> `hasFDerivAt_integral_of_dominated_of_fderiv_le`).  Estimated ~400–600 lines.

**L4 — n-d smoothing injectivity.**
- `laplaceProfile_integrable : Integrable (fun x => exp (-‖x‖/τ))` on
  `EuclideanSpace ℝ ι` via the product domination `≤ ∏ᵢ exp (-|xᵢ|/(√n τ))`
  and `Measure.pi`-Fubini (`volume_euclideanSpace_eq_pi`-transport — verify the
  exact volume-transport lemma name at implementation).
- Subordination: `bernstein_exp_neg_sqrt :
  ∀ r ≥ 0, exp (-r) = ∫ u in Ioi 0, exp (-r²·u) * (4πu³)^{-1/2} * exp (-1/(4u))`
  — one honest 1-d improper-integral computation (substitute `v = 1/(4u)`,
  reduce to `∫₀^∞ e^{-(a v - b/v)²·…}`-type; the standard trick is
  differentiation in `r` + the Gaussian integral, or glasser-style substitution;
  ~150–300 lines; the only genuinely new analysis in L4).
- `fourier_laplaceProfile_pos : 0 < re (𝓕 (fun x => (exp (-‖x‖/τ) : ℂ)) t)` and
  imaginary part `= 0`: Tonelli-swap the subordination against the Fourier
  integral, apply Mathlib's Gaussian Fourier transform
  (`fourierIntegral_gaussian_innerProductSpace`), positivity of the resulting
  `u`-integral.
- Factoring + cancellation: mirror GaussianConvolutionInjectivity.lean:218–288
  verbatim with the Laplace profile (`𝓕 Z_p = charFun_p((-2π)•t)·𝓕 profile`),
  then `Measure.ext_of_charFun`.
- **`laplaceKernelNormalizer_injective_euclidean`** for finite measures;
  instantiate the repo predicate: `laplaceSmoothingInjective_euclidean :
  LaplaceSmoothingInjective (EuclideanSpace ℝ ι) τ` (the predicate from
  LaplaceInjectivity.lean:439 — defined generically, anticipating exactly this).

Status note (2026-07-14).  The profile-integrability and Fourier-cancellation
shell are now implemented in `LaplaceEuclideanInjectivity.lean`:

```
laplaceEuclideanFourierBase_integrable
laplaceKernelNormalizer_injective_euclidean_of_fourier_ne_zero
laplaceSmoothingInjective_euclidean_of_fourier_ne_zero
laplaceKernelNormalizer_injective_euclideanSpace_of_fourier_ne_zero
laplaceSmoothingInjective_euclideanSpace_of_fourier_ne_zero
```

These theorems are axiom-free.  The Euclidean-space versions discharge
integrability of `x ↦ exp (-‖x‖/τ)` by product domination in coordinates.  The
remaining radial Fourier nowhere-vanishing/positivity layer described below has
now also been discharged in `LaplaceRadialFourier.lean`.

**Subordination crux, scalar identity closed (2026-07-14,
`LaplaceRadialFourier.lean`, machine-checked, axiom-free).**  The scalar
Glasser/subordination engine is now implemented:

```
integral_inv_sq_exp_neg_div_sq
integrableOn_inv_sq_exp_neg_div_sq
glasserKernel_integrableOn
integral_inv_sq_mul_glasserKernel_eq_inv_mul_integral
glasserIntegral_eq_mul_integral_inv_sq_mul
hasDerivAt_glasserIntegral
glasserIntegral_zero
hasDerivAt_glasserScaled
glasserScaled_eq_of_pos
tendsto_glasserIntegral_nhdsWithin_zero
tendsto_glasserScaled_nhdsWithin_zero
glasserIntegral_eq_closed_of_pos
subordination_integral_eq_of_pos
exp_neg_eq_subordination_of_pos
```

In words: the reciprocal-Gaussian dominator is integrable; the
self-reciprocal substitution proves
`F(k) = k·∫ x⁻² e^{−x²−k²/x²}`; dominated differentiation gives
`F'(k) = −2F(k)` on `(0,∞)`; the ODE plus the right-limit at `0` gives
`F(k) = (√π/2)e^{−2k}`; and setting `k = a/2` gives the normalized
subordination identity
`e^{−a} = (2/√π)∫_{(0,∞)} e^{−s²}e^{−a²/(4s²)} ds` for `a > 0`.

Status refinement (2026-07-14, Codex pass).  The scalar crux and global Fourier
packaging have now been pushed through to the final `hbase_ne` theorem in
`LaplaceRadialFourier.lean`, still axiom-free:

```
exp_neg_eq_subordination_of_nonneg
fourier_gaussian_sq_norm_eq_real
fourier_gaussian_sq_norm_re_pos
fourier_gaussian_sq_norm_ne_zero
laplaceRadialSubordinationScalar
laplaceEuclideanFourierBase_eq_subordination_integral
laplaceRadialSubordinationScalar_eq_gaussian
laplaceRadialSubordinationGaussianCoeff_pos
laplaceRadialSubordinationGaussianMass_factor
integrableOn_laplaceRadialSubordinationGaussianMass
integral_norm_laplaceRadialSubordination_fourier_slice
integrable_laplaceRadialSubordination_fourier_slice
integrable_laplaceRadialSubordination_fourier_product
laplaceRadialSubordination_inner_fourier_eq
laplaceRadialSubordination_inner_fourier_re_pos
fourier_laplaceEuclideanFourierBase_eq_subordination_integral
integrable_laplaceRadialSubordination_inner_fourier
fourier_laplaceEuclideanFourierBase_re_pos
fourier_laplaceEuclideanFourierBase_ne_zero
laplaceSmoothingInjective_euclideanSpace
```

The closed path is exactly the intended one: product integrability of the
phase-weighted subordination integrand follows from the exact spatial norm
integral and the `s^n e^{-s²}` Gaussian-mass estimate; Fubini identifies the
Fourier transform of the radial Laplace profile with the positive `s`-integral
of positive Gaussian Fourier transforms; strict positivity gives nonvanishing;
and the existing `laplaceSmoothingInjective_euclideanSpace_of_fourier_ne_zero`
gate yields the fully discharged predicate theorem
`laplaceSmoothingInjective_euclideanSpace`.  No new axiom or conditional theorem
has been introduced.

**L5 — radial-measure converse (D2.b).**  *Superseded by the deep-dive plan
§4.9(R8) — kept for the original reasoning; the §4.9 version eliminates the
Δ-layer via the tangential IBP identity, fixes the origin boundary condition
through the `v = r^{n-1}w` substitution, proves half the sign condition, and
isolates the single remaining open point.*
- `IsRadial μ := ∀ (R : LinearIsometryEquiv …), Measure.map R μ = μ` (or
  orthogonal-group orbit formulation); basic transport: `Z_μ`, `ψ_μ` radial,
  atom only at `0`.
- Classical C²: `∫ dμ(y)/‖x-y‖ < ∞` locally uniformly on `ℝⁿ∖{0}` for radial
  `μ` (`n ≥ 2`; sphere-decomposition estimate), then twice-dominated
  differentiation ⟹ `Z_μ ∈ C²(ℝⁿ∖{0})` with
  `ΔZ_μ(x) = ∫ Δk(x-y) dμ(y)`, `Δk(u) = (1/τ² - (n-1)/(τ‖u‖))·k(u)`.
- Kernel ODE (♦) pointwise off `0`:
  `ψ - τ²Δψ = (n+1)τ²·Z` (from `matern32Profile` radial computation
  `g'' + (n-1)g'/r = (‖·‖-part)`, all elementary `deriv` algebra).
- Radial reduction: `Z̃(r) := Z(r•e₁)` etc.; `∂ᵣ`-versions of the above;
  zero drift ⟹ `m̃ := ψ̃'/Z̃` shared; Wronskian `w := Z̃_q·Z̃_p' - Z̃_p·Z̃_q'`;
  derive the scalar system
  `2m̃'w + m̃(w' + (n-1)w/r) = -(n+1)w` on `{r > 0}` (classical Leibniz from the
  C² layer — the (★) derivation specialized to one variable).
- Boundary: `w(0⁺) = 0` (C¹ radial ⟹ `∇Z(0) = 0`; needs no-atom-at-`0` case
  split — with atoms at `0`, use L3 to align them first and strip), `w(∞) = 0`
  (both factors decay; quantitative bound from `Z ≤ 1`, `|∇Z| ≤ 1/τ·Z`-style).
- Propagation: reuse the Frontier-A abstract interval machinery
  (`exists_Ioo_linear_bound_of_hasDerivAt_zero`,
  `intervalPrimitive_hasDerivWithinAt_Ici_of_rightContinuous`,
  `intervalIntegrable_of_measurable_bounded` are measure-generic; the K/m/W
  concrete lemmas serve as templates only).  Grönwall on maximal intervals of
  `{m̃ ≠ 0}` shooting from `0⁺` and `∞`; trichotomy across zeros of `m̃`.
- **Open lemma (radial L3)**: `(r + m̃(r))' ≥ 0` — attack via the cylindrical
  conditioning + reflection pairing of §4.6(e); *decision point*: if unproved
  after the numerics check, ship the theorem with it as a named hypothesis
  (`laplaceZeroDrift_identifies_of_radial_of_monotoneCentroid`) — still a
  legitimate conditional milestone — and keep the unconditional form as the
  target.
- Endgame: `w ≡ 0 ⟹ (Z_p/Z_q)' = 0 ⟹ Z_p = c·Z_q ⟹ c = 1` (masses:
  `∫ Z = ‖k‖₁` Tonelli or the far-field limit) `⟹ p = q` by L4.
- Headline: `laplaceZeroDrift_identifies_of_radial (hp : IsRadial p)
  (hq : IsRadial q) : p = q` + audit entries.

**Numerics side-quests (external to Lean, before/parallel to L5):**
(i) centroid-monotonicity conjecture test (§4.6(e)) — decides L5's shape;
(ii) drift-residual minimization over small atomic configurations in ℝ² —
falsification screen for the ℓ² conjecture itself.  Both are ~50-line scripts;
record outcomes in this file.

### 4.9 Radial deep-dive (third pass, 2026-07-14): closing L5's gaps

*This pass resolved most of L5's open design questions by hand-derivation.
Everything below is checked symbolically (per-shell identities verified by
direct IBP; all formulas re-derived twice); the single remaining open
mathematical point is isolated in (R5).*

**(R1) Cylindrical reduction and exact first-order formulas.**  Fix the ray
direction `e₁`; write `t = ⟨y,e₁⟩`, `ρ = ‖y - t·e₁‖`, `X = t - r`,
`d = √(X² + ρ²)`, `w = e^{-d/τ}`, and `E_w[·]` for the `w`-weighted mean under
`p`.  Along the ray (`x = r•e₁`), with `Z̃(r) = Z_p(r•e₁)` etc.:

```
Z̃'(r)  = (1/τ)·∫ w·(X/d) dp            (radial component of ∇Z)
D̃'(r)  = -Z̃ + (1/τ)·∫ w·(X²/d) dp
m̃' + 1 = (1/τ)·( E_w[X²/d] - E_w[X]·E_w[X/d] ) = (1/τ)·Cov_w(X, X/d).
```

At `n = 1`, `X/d = sgn X` and `Cov(X, sgn X) = E|X| - E[X]E[sgn X] ≥ 0` is a
one-line proof of the 1-d L3 — so `Cov_w(X, X/d)` is the *correct* radial
generalization of the tilted-mean coefficient, and everything below is about
lower-bounding it.

**(R2) The tangential IBP identity (T) — the Laplacian eliminated.**  For every
radial `μ` (probe at `r > 0`):

```
(T)      ∫ (ρ²/d)·w dμ  =  ((n-1)τ/r) · ∫ t·w dμ .
```

*Proof per shell of radius `s`* (numerators are linear in `μ`, so shells
suffice): parametrize by `u = cosθ ∈ [-1,1]`, `d = √(r²+s²-2rsu)`,
`ρ² = s²(1-u²)`, shell density `∝ (1-u²)^{(n-3)/2}`; one 1-d integration by
parts, using `∂_u e^{-d/τ} = (rs/(τd))e^{-d/τ}` and
`∂_u (1-u²)^{(n-1)/2} = -(n-1)u(1-u²)^{(n-3)/2}` (boundary terms vanish).
Verified independently for `n = 2` shells.  Consequences (with `c(r) = r + m̃`
the centroid radial component, `d = X²/d + ρ²/d` exactly):

```
E_w[ρ²/d] = (n-1)τ·c(r)/r ,        E_w[d] = E_w[X²/d] + (n-1)τ·c(r)/r .
```

**(T) is the radial form of (♦), obtained with no second derivatives** — in
Lean this replaces the entire `Δ`-layer for L5: the C² machinery is *not
needed* for the identities, only 1-d IBP under the integral.  (The C² layer is
still how `m̃'` exists classically; see (R6).)

**(R3) The `v`-substitution: the radial system is exactly 1-d-shaped.**  Set
`v(r) := r^{n-1}·w(r)` with `w = Z̃_q Z̃_p' - Z̃_p Z̃_q'`.  The geometric
`(n-1)/r` term is absorbed:

```
m̃·v' = -( (n+1) + 2m̃' )·v ,      K := τ·m̃·v ,      K' = -τ·( m̃' + (n+1) )·v .
```

This is the proven 1-d system `{m W' = -(2 + 2m')W, K = τmW, K' = -τ(m'+2)W}`
with the constant `2 ↦ n+1`.  Defining `μDeriv_rad := m̃' + (n-1)`, the
K-equation reads `K' = -τ(μDeriv_rad + 2)·v` — **the exact shape consumed by
the abstract propagation layer** (`abel_right_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_upper`
and siblings in LaplaceACPropagation.lean, applied at
LaplaceUnconditionalConverse.lean:699/:782).  Recon of the 1-d trichotomy
(`laplaceCompanionAlignmentDefect_eq_zero_of_zeroDrift_gen`, :882–:1010)
confirms the layer consumes exactly four inputs and never unfolds
`μDeriv = m'`:
1. the global lower bound `hμ1 : ∀ t, μDeriv t + 1 ≥ 0` (:888);
2. compact-interval bounds on the Abel coefficient `(μDeriv+2)/m` inside
   `{m ≠ 0}` (`exists_uIcc_bound_laplaceGapCoeff`, :553 — pure continuity, no
   sign);
3. differentiability of `m` at its zeros + the generic linear-bound helper
   (`exists_Ioo_linear_bound_of_hasDerivAt_zero`, :695/:778);
4. the identity `K = τ·m·(W-slot)` with `W-slot` bounded, for edge vanishing.

For the radial port, item 1 becomes `m̃' + n ≥ 0` — see (R5); items 2–4 are
*easier* than 1-d (classical derivatives, (R6)).  One caveat: the abstract
lemmas hard-code the pair `(μDeriv+1, μDeriv+2)` in two roles (the `v`-equation
coefficient is `2(m̃' + (n+1)/2) = 2μDeriv_rad + (3-n)`, not `2(μDeriv_rad+1)`,
for `n ≠ 1`); if any lemma uses *both* shapes simultaneously, generalize its
constants to a symbolic pair — a mechanical edit of LaplaceACPropagation.lean
(the lemmas are interval-abstract already).

**(R4) Boundary behavior — both ends improve on 1-d.**
- **`v(0⁺) = 0 unconditionally** (`n ≥ 2`): even when `p, q` have atoms at the
  origin (the only possible atoms of a radial measure), `w(0⁺)` is finite
  (`= [q₀Z̃_p(0) - p₀Z̃_q(0)]/τ`, generally ≠ 0!) but `r^{n-1}` kills it.  The
  earlier idea "align origin atoms via L3 first" is unnecessary *and* wouldn't
  work (`D(0) = 0` for radial measures makes the L3 relation vacuous at `0`);
  the `v`-substitution resolves it outright.  Moreover `m̃(0) = 0` and `m̃` is
  Lipschitz at `0`, so `|K| ≤ C·rⁿ` near `0`: the origin behaves exactly like
  an interior `m`-zero edge (coefficient `~ 1/r`, super-linear `K`-bound) and
  the *same* abel edge lemma applies — `r = 0` is a universal edge.
- **Ray at `∞`:** the fundamental solution `exp ∫ c_v` with
  `c_v ≥ 1/|m̃| ≥ 1/(C + n·r)` grows like `r^{1/n}` (using `m̃' ≥ -n` for the
  linear bound on `|m̃|`), so `K ≡ 0` on the unbounded interval follows from
  `K(r_k) → 0` along *any subsequence*.  Since `|D̃_p| ≤ τ/e` always,
  `|K| ≤ (τ²/e)·r^{n-1}·Z̃_q`-scale, and `r^{n-1}·Z̃_p·Z̃_q → 0` holds whenever
  both measures have finite `(n-1)/2`-moments (split `‖y‖ ≶ r/2`; Markov).
  **L5 v1 therefore carries the mild hypothesis: finite `(n-1)/2`-moments**
  (n = 2: half-moment; n = 3: first moment; automatic for compact support).
  Removing it is a refinement, not a blocker (only a liminf along a subsequence
  is needed).

**(R5) The sign condition — mostly proved, one open case.**  The port needs
`m̃' + n ≥ 0` pointwise, i.e. by (R1)+(T) the **radial slack inequality**

```
(RSI)      E_w[X²/d] + (n-1)τ  ≥  m̃ · E_w[X/d] .
```

Status:
- `n = 1`: RSI *is* the proven 1-d L3 (`E|X| ≥ E[X]E[sgn X]`).  ✓
- **`m̃ ≤ 0` (centroid at or behind the probe): PROVED.**  Cauchy–Schwarz gives
  `m̃² = E[√d·(X/√d)]² ≤ E_w[d]·E_w[X²/d]`, and with
  `E_w[d] = E_w[X²/d] + (n-1)τc/r` (T) one gets
  `|m̃|·|E[X/d]| ≤ √(Q(Q + (n-1)τ·c/r)) ≤ Q + (n-1)τ` for `c ≤ 2r`
  (`Q := E_w[X²/d]`), which covers all of `m̃ ≤ 0` (`c ≤ r`).  ✓
- **At zeros of `m̃`: FREE and strict.**  `m̃ = 0` kills the quadratic term:
  `m̃' + 1 = (1/τ)E_w[X²/d] ≥ 0 > -1`, no inequality needed.  (This is why the
  edge-crossing steps of the trichotomy are unconditionally safe.)  ✓
- **`m̃ > 0` (centroid beyond the probe): OPEN.**  Pair form: RSI ⟺
  `∬ [ (X-X')(X/d - X'/d') + 2(n-1)τ ]·w w' dp dp' ≥ 0`.  The bracket is ≥ 0
  for same-`ρ` pairs (both factors co-monotone in `X`) but can be negative
  across scales (`X = ε, ρ = 0` vs `X' = L, ρ' ≫ L`); such configurations are
  constrained by the shell structure (each shell spreads over the whole
  `(X,ρ)`-arc) and by the `2(n-1)τ` buffer, which is exactly what the 1-d case
  lacks (`n = 1` has buffer `0` and is still true!).  **This is now the single
  open mathematical point of L5.**  Numerics spec: sample shell mixtures
  (2–4 shells, random radii/masses), scan `r`, compute
  `Q + (n-1)τ - m̃E[X/d]` by quadrature; also directly scan `m̃' + n` via finite
  differences.  If violations exist, weaken to the interval-version needed by
  the abel layer (the bound is only consumed on `{m̃ ≠ 0}` up to its edges —
  near the edges it is free by the zero-case above, so a violation must be
  *interior and quantitatively large* to break the argument; the layer's
  Grönwall survives `μDeriv ≥ -1 - δ` at the cost of edge-exponent margins,
  which the `n ≥ 2` constants have — see (R3): the K-coefficient margin is
  `m̃' + n + 1 ≥ 1` under `m̃' ≥ -n`, and the same argument runs with any
  uniform `m̃' ≥ -n - 1 + ε`).  Fallback (unchanged): ship
  `laplaceZeroDrift_identifies_of_radial_of_slack` with RSI as a named
  hypothesis.
- Bonus theorem (independent of RSI): the tangential monotonicity
  `c(r) ≥ 0` for all radial `p` — reflection-pairing across the hyperplane
  `⟂ e₁` (nearer point gets the larger weight).  Worth proving in L5 as
  `radialCentroid_nonneg`; it feeds the `|m̃| ≤ r + …` bounds.

**(R6) Regularity and the `n = 2` caveat.**  For radial `μ` and `n ≥ 3`,
`∫ dμ(y)/‖x-y‖` is locally uniformly finite off the origin (shell integrand
`θ^{n-2}/d ~ θ^{n-3}` integrable), so `Z̃, ψ̃, m̃` are classically `C¹`/`C²` on
`(0,∞)` and *two-sided* derivatives can be used throughout — none of the 1-d
one-sided contortions.  For `n = 2` the shell-through-probe integral is
log-divergent: `m̃'` fails to exist classically *at radii carrying shell mass*,
but the divergence is logarithmic, hence `m̃'` is still locally *integrable*
and the propagation can run in FTC form (absolutely-continuous `K` with
integrable derivative) or via the 1-d one-sided port.  **Decision: L5 v1
targets `n ≥ 3`; the `n = 2` variant is v2** (same skeleton, càdlàg/FTC layer
swapped in — the 1-d file is the template for exactly this).

**(R7) Statement design: define radial measures by shell mixtures.**  To avoid
building rotation-orbit disintegration (Haar-averaging `∫ G(Ry) dHaar(R)` =
sphere average — the projection-density fact is *not* in Mathlib), state L5
for measures *given* as shell mixtures:

```
radialMixture (μ̃ : Measure ℝ≥0) : Measure E := μ̃.bind (fun s => uniformShell s)
```

with `uniformShell s` the normalized sphere measure (build from Mathlib's
`Measure.toSphere` / HaarToSphere infrastructure, or as the pushforward of the
`s`-dilation of the unit-sphere measure).  Disintegration then holds *by
definition* (`Measure.bind` + `integral_bind`), the per-shell IBP (T) applies
directly, and no group theory enters.  This loses no honest generality — shell
mixtures *are* the rotation-invariant measures — and the equivalence
(`rotation-invariant ⟹ shell mixture`) can be added later as an independent
lemma if wanted (it, not L5, is where Haar orbit-uniqueness would be needed).
Atoms of `μ̃` are exactly shell masses; an atom of `μ̃` at `0` is the origin
atom.

**(R8) Updated L5 lemma plan (supersedes the L5 block of §4.8).**
File `LaplaceRadialMeasureConverse.lean`, `n ≥ 3` (v1):
1. `uniformShell`, `radialMixture`, `integral_radialMixture`
   (`Measure.bind` API; probability instances).
2. Ray objects `Z̃, D̃, ψ̃, m̃, c` as 1-d functions; symmetry lemmas
   (`D` tangentially zero; `m̃(0) = 0`; `radialCentroid_nonneg`).
3. First-derivative layer: dominated differentiation in `r` (integrands
   Lipschitz in `r` with global `1/τ`, `τ/e` constants) → formulas (R1).
4. (T) via per-shell 1-d IBP + `integral_bind`; corollaries
   `E_w[ρ²/d] = (n-1)τc/r`, `E_w[d] = Q + (n-1)τc/r`.
5. Second-derivative layer (`n ≥ 3`): local uniform `∫dμ/d < ∞`, `m̃'`
   classical; `m̃' + 1 = Cov_w(X, X/d)/τ` (quotient rule + (R1)).
6. Sign layer: RSI for `m̃ ≤ 0` (Cauchy–Schwarz, step 4), the zero-case
   identity, and either the `m̃ > 0` proof (pending (R5)) or the named
   hypothesis.
7. `v`-system: `w`, `v = r^{n-1}w`, `K = τm̃v`; product/quotient calculus;
   `K'` formula; `|K| ≤ C·rⁿ` near `0`; `K → 0` at `∞` under
   `(n-1)/2`-moments.
8. Propagation: instantiate (or symbolically generalize, (R3)) the
   LaplaceACPropagation abel lemmas with `μDeriv_rad = m̃' + (n-1)`; trichotomy
   over the zero set of `m̃` on `(0,∞)` with the `r = 0` universal edge —
   mirror :882–:1010 of the 1-d file.
9. Endgame: `K ≡ 0 ⟹ m̃·v ≡ 0 ⟹ (Z̃_p/Z̃_q)' = 0` on `{m̃ ≠ 0}`-components +
   continuity across zeros (`w = 0` a.e. ⟹ ratio locally constant everywhere:
   same gluing as 1-d) ⟹ `Z_p = c·Z_q` radially ⟹ everywhere (radiality) ⟹
   `c = 1` (Tonelli mass or `r → ∞`) ⟹ `p = q` by **L4**.
10. Headline `laplaceZeroDrift_identifies_of_radialMixture` (+ `_of_slack`
    variant if (R5) stays open); audit entries.

Dependencies: L0 (potential layer optional here — (T) replaces it radially),
L4 (final step).  L3 not required.  Revised effort: 4–6 sessions for v1;
`n = 2` v2 +2–3 sessions; risk now concentrated in (R5) `m̃ > 0` and the
`uniformShell` infrastructure (Mathlib sphere-measure API maturity — recon
`Measure.toSphere` before starting).

---

## 5. Recommended sequencing

*(Revised 2026-07-14 after verifying the paper's Eq. (12) uses the ℓ² norm —
the ℓ² track is the paper-faithful target and now leads.)*

1. **Next session: L0 (foundations) + start L1** — `ψ`, `D = ∇ψ`, potential
   alignment, then the moment-minimal far-field limits.  Everything here is
   low-risk and load-bearing for all later milestones.
2. **L1 finish + L2** (far-field theorem `zeroDrift_tiltedCentroid_eq`;
   dimensional reduction + the collinear corollary — the first unconditional
   n-d ℓ² statement).
3. **L3 (atom alignment)** — the new n-d theorem
   `q({a})·D_p(a) = p({a})·D_q(a)`, arbitrary measures, `n ≥ 2`; publishable
   structure on its own and required by any eventual endgame.
4. **L4 (n-d smoothing injectivity)** — the reusable final step; one genuinely
   new integral computation (Bernstein subordination), the rest mirrors the
   Gaussian template.
5. **Numerics side-quests** (before/parallel to L5): the RSI `m̃ > 0` case
   (§4.9(R5)) and the drift-residual falsification screen — they decide L5's
   final shape.
6. **L5 (radial-measure converse, D2.b)** — implement per §4.9(R8) (`n ≥ 3`
   first): the sign condition is already proved for `m̃ ≤ 0` and at zeros;
   unconditional if RSI's `m̃ > 0` case falls, otherwise ship the
   named-hypothesis form `…_of_slack`.
7. **D2.c/D2.d paper-first**, informed by L3/L5 and the numerics.
8. **Opportunistic: implement §3 (Finding 1, ℓ¹).**  Deliverables:
   `productKernel_zeroDrift_identifies` (meta-theorem),
   `productLaplaceZeroDrift_identifies` (ARD bandwidths),
   `laplaceZeroDrift_identifies_l1` (PiLp-1 headline parity),
   `productLaplaceSmoothingInjective`, mixed Laplace–Gaussian corollary,
   legitimacy surface, audit entries.  Fully derisked; not paper-faithful for
   `n ≥ 2`; good filler between ℓ² milestones or a warm-up for pi-measure
   infrastructure.

Effort guess: L0–L2 ≈ 2–3 sessions; L3 ≈ 1–2; L4 ≈ 1–2; L5 ≈ 3–5 (gated on the
radial-L3 lemma or its named-hypothesis fallback); ℓ¹ track ≈ 1–2; D2.c/D2.d
open-ended.  Full lemma-level detail for every L-milestone: §4.8.

---

### 4.10 L5 v1 implementation architecture (fourth pass, 2026-07-15): the `n = 3` explicit route

*This pass supersedes the (R8) execution plan for v1.  L0–L4 are all closed
(L4 = `laplaceSmoothingInjective_euclideanSpace`, LaplaceRadialFourier.lean:990,
fully unconditional).  Re-deriving the radial system against the actual 1-d
codebase (LaplaceGeneralConverseCompanionWronskian.lean,
LaplaceUnconditionalConverse.lean:882–1010, LaplaceACPropagation.lean) exposed
a dramatically cheaper architecture at `n = 3` in which every hard analytic
layer of (R8) — sphere-measure API, zonal density, tangential IBP, the C²/Δ
layer, `∫dμ/d` estimates — either dissolves into elementary closed-form algebra
or is not needed at all.  Everything below is hand-verified (each identity
re-derived twice, cross-checked against (R1)–(R4) and the 1-d code).*

**Status log (append entries here as work proceeds; this section is the
session-handoff spec).**

- 2026-07-15: architecture derived and recorded; nothing implemented yet.
  Next actions: (i) RSI numerics for the `m̃ > r` regime, (ii) S-layer file
  `LaplaceRadialShell3.lean`.
- 2026-07-15 (numerics, same session): **RSI robustly TRUE empirically; the
  open `m̃ > r` case looks strictly safe.**  C# Simpson scan (script:
  scratchpad `RsiScan.cs`; τ=1 wlog): 800 random 1–4-shell configs × 90
  probes (72k samples) + 14k adversarial near+far 2-shell samples.  The
  finite-difference check validates `m̃' = -1 + Cov_w(X, X/d)/τ` to 1e-10
  (independent confirmation of the (F3)/(F6) formula derivation).  Results:
  in the OPEN region `m̃ > r`, `Cov ≥ +0.04τ > 0` always (min over both
  scans; adversarial min `+1.25τ`) — consistent with the within-shell
  co-monotonicity argument (`s > r` shells have `Cov_within ≥ 0`).  Worst
  case OVERALL is a *single shell just behind the probe* (`s/r ≈ 0.8`,
  `m̃ < 0` region): `Cov ≈ -0.0457τ` — a 40× margin against the needed
  `-2τ`, and it appears mixtures only improve on the single-shell worst
  case.  So `RadialSlack₃` ships as a conjectured-true named hypothesis;
  identified route to REMOVE it later (v2): law of total covariance over
  the shell mixture + within-shell Chebyshev (`s ≥ r` shells free) +
  explicit single-shell bound `Cov_shell ≥ -τ/8`-ish for `s < r` (closed
  forms exist) + cross-shell mean-monotonicity — all elementary but gnarly;
  not v1.
- 2026-07-15 (S-layer, part 1 — **committed, green**): `LaplaceRadialShell3.lean`
  created.  Done: `sphereChart`/`rayProbe` + component `rfl` lemmas;
  `continuous_sphereChart` (via `continuous_pi` + `change` + `fun_prop`, and
  `PiLp.continuous_toLp` found by `fun_prop`); `sphereChart_normSq = 1`;
  **`dist_sq_rayProbe_smul_sphereChart`** (`‖r•e₁ - s•Φ(u,φ)‖² = r²+s²-2rsu`,
  the φ-independence heart); `chartBase` (+ `IsProbabilityMeasure`, mass
  `4π·(4π)⁻¹ = 1`); `chartMap`, `radialMixture₃` (+ `IsProbabilityMeasure` via
  `Measure.isProbabilityMeasure_map`); **`integral_radialMixture₃`** (master
  collapse `∫ f d(radialMixture₃ ν) = ∫ s ∫ w f(chartMap(s,w)) dchartBase dν`
  via `integral_map` + `integral_prod`).  Gotchas recorded: `μ̃` (μ+combining
  tilde) is NOT a valid Lean identifier — use `ν`; `ℝ≥0∞` needs `open scoped
  ENNReal`; `.comp` needs explicit `(f := …)`; `Matrix.cons_val_two` is `rfl`.
  Remaining S-layer: zonal φ-collapse `∫ w ∂chartBase → (1/2)∫_{-1}^1`, inner
  `⟪s•Φ,e₁⟫ = su`, ray-object distance forms, T₃ identity, ray-mass identity.
- 2026-07-15 (S-layer parts 2–3 — **committed, green**): `integral_chartBase_zonal`
  (the φ-collapse); `shellDist`, `shellZ`, `shellC`;
  `norm_rayProbe_sub_smul_sphereChart`; kernel/companion continuity + nonneg +
  bounds (`≤ 1`, `≤ τ` via `1+x ≤ eˣ`); distance-collapse chart lemmas;
  **`kernelNormalizer_radialMixture₃`** (`Z̃_ν(r) = ∫ shellZ τ r s dν`) and
  **`kernelNormalizer_companion_radialMixture₃`** (`C̃_ν(r) = ∫ shellC τ r s dν`).
  So Z̃ and C̃ ray objects have clean ν-mixture closed forms.  More gotchas:
  `setIntegral_const` gives `μ.real s` (use `Real.volume_real_Ioc_of_le`);
  `rw [← Real.exp_zero]` clobbers the `1` in `1/τ` (rewrite in a hyp instead);
  `HasFiniteIntegral.of_bounded` + `Measure.isProbabilityMeasure_map` (dot forms);
  `field_simp [hτ.ne']; ring` for `τ(d/τ+1)=τ+d`.  **A full session-handoff doc
  now exists: `LaplaceL5_HANDOFF.md`** (self-contained; new agents start there).
  Remaining S-layer: drift e₁-component `shellD` (needs the eval-CLM /
  `ContinuousLinearMap.integral_comp_comm` to extract component 0 of the vector
  drift numerator), T₃ identity, ray-mass identity.
- 2026-07-15 (R-layer C¹ — **committed, green; THE C¹ RISK ITEM IS CLOSED**):
  `LaplaceRadialRay3.lean` now has the full C¹ layer.  Pointwise:
  `norm_rayProbe_sub_eq_sqrt`, junk-safe bounds (`|X|≤d`, `|X/d|≤1`, `X²/d≤d`),
  `hasDerivAt_norm_rayProbe_sub` (via `HasDerivAt.sqrt`), and the three kernel
  probe-derivatives (`hasDerivAt_laplace{,Companion}Kernel_rayProbe'`,
  `hasDerivAt_laplaceKernel_mul_coord'`), all conditioned only on
  `‖rayProbe r − y‖ ≠ 0`.  Measure: `radialMixture₃_ae_probe_ne` (the collision
  set is null — its chart preimage sits in the `u = 1` chartBase-null slice).
  Dominated differentiation (`hasDerivAt_integral_of_dominated_loc_of_deriv_le`
  with `s := Ioi 0` via `Ioi_mem_nhds`, global constant dominators `1/τ`,
  `e⁻¹`, `e⁻¹+1`): **`hasDerivAt_radialRayZ₃`** (`Z̃' = (1/τ)·radialRayZd₃`),
  **`hasDerivAt_radialRayC₃`** (`C̃' = (1/τ)·D̃` — the closure's first leg),
  **`hasDerivAt_radialRayD₃`** (`D̃' = (1/τ)·radialRayQ₃ − Z̃`).  New y-level
  payload objects `radialRayZd₃ := ∫(X/d)w̄`, `radialRayQ₃ := ∫(X²/d)w̄`;
  helper `radialRayD₃_eq_integral_coord` (component 0 commuted through the
  Bochner integral via `EuclideanSpace.proj` + `integral_comp_comm` +
  `EuclideanSpace.coe_proj`).  Lean gotchas: `HasDerivAt.sqrt` exists (never
  fight `.comp` with `√·`); value-massaging via
  `have hval : v₁ = v₂ := by ring` then `rw [hval] at h` is far more robust
  than `convert h using 1`; `not_imp` is ambiguous (use `by_contra` + direct
  λ-construction); `push_neg` is deprecated here (`not_le.mp`/`not_forall.mp`);
  `field_simp` needs explicit `[hne, hτ.ne']`; `mul_div_mul_left _ _
  two_ne_zero` cleans `2a/(2b) = a/b`.  Remaining R-layer: T̃/P y-level bridges
  (P needs a measurable-G variant of `integral_chartBase_zonal` for the `ρ²/d`
  collision), closure identity, sign layer.
- 2026-07-15 (closure identity — **committed, green**): `LaplaceRadialRay3.lean`
  now proves **`radialRayC₃_closure`**: `C̃ = Q̃ + 3τ·Z̃ + (2τ/r)·D̃` on the
  open ray for radial mixtures supported on `[0,∞)` — together with
  `hasDerivAt_radialRayD₃` this is exactly the F4 closure
  `C̃ = τD̃' + 4τZ̃ + (2τ/r)D̃`.  Supporting chain, all new and green:
  `norm_rayProbe_sub_sq_eq` (`d² = X² + ρ²` in coordinates);
  `rhoSq_div_norm_rayProbe_le` + `abs_laplaceKernel_mul_rhoSq_div_le` (`ρ²/d ≤
  d`, `K·ρ²/d ≤ τe⁻¹`, junk-safe); **`integral_chartBase_zonal_measurable`**
  (the measurable-G φ-collapse, needed because the `ρ²/d` chart section is
  discontinuous at the collision `s = r, u = 1`); the two y-level bridges
  **`radialRayT₃_eq_integral`** (`T̃ = ∫K·y₀`) and
  **`radialRayRhoSqOverDist₃_eq_integral`** (`P̃ = ∫K·(ρ²/d)`); the **mixture
  T₃** `radialRayRhoSqOverDist₃_eq_T` (`P̃ = (2τ/r)·T̃`; per-shell T₃ + zero
  contribution of the `s = 0` origin atom via `shellT_zero_right`/
  `shellRhoSqOverDist_zero_right` + `radial_ae_nonneg` from `hsupp`); and the
  axial split `radialRayT₃_eq_D_add_Z` (`T̃ = D̃ + r·Z̃`).  Gotchas: `abs_add`
  → `abs_add_le` in this Mathlib; `div_add_div_same` does not exist (use
  `← add_div`); `pow_two` not `sq` as a rewrite; `LE.le.eq_or_gt` dot-resolves
  into `Real.le` (use `eq_or_lt_of_le`); `nlinarith` cannot prove equalities
  (use `linear_combination`); collision branches close with
  `simp only [h0, div_zero, mul_zero, add_zero]` (rw fights beta-reduction).
  Remaining R-layer: sign layer (`Z̃ > 0`, `m̃`, C–S `m̃ ≤ r` case, zero case,
  `RadialSlack₃`), then the system file.
- 2026-07-15 (sign layer — **committed, green; THE R-LAYER (5b) IS COMPLETE**):
  `LaplaceRadialRay3.lean` closes the sign layer.  **Discovery: no
  Cauchy–Schwarz needed** — the C–S step of (F6) reduces to the pointwise
  AM–GM `2·K·|X| ≤ K·(X²/d) + K·d` (i.e. `(|X|−d)² ≥ 0`, `two_mul_abs_coord_le`,
  junk-safe), integrated to **`abs_radialRayD₃_le`**:
  `|D̃| ≤ Q̃ + (τ/r)(D̃ + rZ̃)` via the T₃-corollary `integral_kernel_mul_norm_eq`
  (`S̃ = Q̃ + (2τ/r)(D̃ + rZ̃)`).  Also green: `radialRayZ₃_pos` (support = univ
  + `integral_pos_iff_support_of_nonneg`); `abs_radialRayZd₃_le` (`|Z̃d| ≤ Z̃`);
  `radialRayM₃`/`radialRayMDeriv₃` defs; `hasDerivAt_radialRayM₃` (statement
  rewritten by `rfl`-function-eq then `exact .div` — value is defeq);
  `radialRayMDeriv₃_cov` (`τ(m̃'+1)Z̃² = Q̃Z̃ − D̃Z̃d`);
  **`radialRayMDeriv₃_ge_of_le`** (`m̃ ≤ r ⟹ m̃' ≥ −3`, the PROVED sign case);
  `RadialSlack₃` (the named far-tilt hypothesis, `∀r>0, r < m̃ r → −3 ≤ m̃' r`);
  `radialRayMDeriv₃_ge` (combined).  Gotchas: `ENNReal.zero_lt_one` doesn't
  exist (plain `zero_lt_one`); `le_or_lt` → `le_or_gt`; `rw [hfe] at h` on a
  `HasDerivAt` function slot fails (use `rfl`-function-equation on the GOAL);
  after `integral_congr_ae … fun y => ?_` use `simp only`, never `rw`
  (beta-reduction).  NEXT: `LaplaceRadialSystem3.lean` (K̂ system F5,
  trichotomy F7, endgame steps 1–3), the ray-mass identity, invariance file,
  converse assembly.
- 2026-07-15 (system file started — **committed, green**):
  `LaplaceRadialSystem3.lean` created with the zero-drift foundation:
  `zeroDrift_ray_meanShift_eq` (ray evaluation of `ZeroDrift`),
  **`zeroDrift_ray_D_mul_Z`** (`D̃_pZ̃_q = D̃_qZ̃_p` — component 0 of the
  mean-shift equality through the bridges; note the `rfl`-haves folding the
  raw `∫ k•(y−x)` into `laplaceWeightedDisplacement` before the bridge
  rewrites), `zeroDrift_ray_M_eq` (common `m̃`), `zeroDrift_ray_MDeriv_eq`
  (common `m̃'` via `HasDerivAt.unique` on eventually-equal functions),
  **`zeroDrift_D_deriv_bridge`** (the covariance bridge: the two
  representations of `D̃'` — the dominated-differentiation value
  `(1/τ)Q̃−Z̃` and the product value `m̃'Z̃ + m̃(1/τ)Z̃d` — agree by
  uniqueness; NO division algebra), the system objects
  `radialRayW₃/V₃/K₃/Khat₃`, and **`zeroDrift_C_eq`** (the fully-substituted
  first-order form `C̃_ν = τm̃'Z̃_ν + m̃Z̃d_ν + 4τZ̃_ν + (2τ/r)m̃Z̃_ν`, the
  `g_ν`-form).  **The remaining K̂-theorems are pure `ring` algebra over
  `zeroDrift_C_eq`** (see the paper derivation in this log): `K̂ = τm̃v` and
  `hasDerivAt K̂ = −τ(m̃'+4)v` — the HasDerivAt for K̂ assembles from the
  product rule on `r²(C̃_pZ̃_q − C̃_qZ̃_p)` with the four R-layer derivative
  theorems, then rw [g_p, g_q, D=MZ-facts]; field_simp [hr.ne']; ring.
  Gotchas: section-variable instances unused in a lemma are hard errors
  (`omit [...] in`); the repo linter forbids goal-changing `show` (use
  `change`); beta-reduction under `EventuallyEq` needs `change` before `rw`.
- 2026-07-15 (system layer COMPLETE — **committed, green**):
  `LaplaceRadialSystem3.lean` now carries the full F5/F8 system:
  **`radialRayKhat₃_eq_M_mul_V`** (`K̂ = τm̃v`) and
  **`hasDerivAt_radialRayKhat₃`** (`K̂' = −τ(m̃'+4)v`) — proven by assembling
  the R-layer derivative theorems and reducing with the `g_ν`-form; the
  coefficient matches the 1-d wrappers' `2μ̂/m̃` with `μ̂ = (m̃'+4)/2`.
  Boundary: **`abs_radialRayKhat₃_le`** (`|K̂| ≤ τr²`, the r = 0 universal
  edge) and **`tendsto_radialRayKhat₃_atTop`** (`K̂ → 0` under first moments)
  via: `tendsto_mul_measureReal_Ici_atTop` (the tail lemma, dominated
  convergence on the truncated first moment), per-shell decay
  (`shellZ_le_exp`, `shellC_le_matern` with `matern_antitone` from
  `1+x ≤ eˣ`), `continuous_shellZ/C` (via `continuous_of_dominated`), the
  mixture split bounds `radialRayZ₃_le_split`/`radialRayC₃_le_split`, a
  pointwise envelope (`abs_radialRayKhat₃_le_envelope`) and
  `tendsto_rpow_mul_exp_neg_mul_atTop_nhds_zero` + `squeeze_zero_norm'`.
  Gotchas: `set x := e with h` REWRITES `e` in all hypotheses (a later
  `rw [← h]` then fails — already folded); `simpa [Function.comp]` on a
  `.comp`-tendsto can fail (`h.congr fun r => rfl` is robust);
  `ENNReal.toReal_one` (not `one_toReal`); the repo linter forbids flexible
  `simp at h` (name the lemmas or use `rw`); `hasDerivAt_pow 2 r` + `simpa`
  for `(x²)' = 2r`.  NEXT: the trichotomy (F7 — mirror
  `LaplaceUnconditionalConverse.lean:882–1010` with `LaplaceACPropagation`
  lemmas), endgame steps 1–3, ray-mass identity, invariance, headline.
- 2026-07-16 (TRICHOTOMY + RAY ENDGAME COMPLETE — **committed, green**):
  `LaplaceRadialConverse3.lean` now proves, in order: the continuity layer
  (`continuousAt_radialRay{Q,Zd,MDeriv,M,V}₃`), the Abel shape
  (`hasDerivAt_radialRayKhat₃_abel`), the three edge cores
  (`radialRayKhat₃_eq_zero_on_Ioo_of_{left,right}Edge`, `…_on_ray` — the
  1-d propagation wrappers consume the radial system with an FTC primitive,
  since the coefficient is genuinely continuous here), the origin-edge
  providers (`shellD_zero_left`, `radialRayD₃_zero`, `abs_radialRayQ₃_le`,
  global `continuous_radialRay{Z,D}₃`, **`abs_radialRayD₃_le_linear`**
  (`|D̃(t)| ≤ (e⁻¹+1)t` via MVT + ε→0 `ge_of_tendsto`),
  **`radialRayM₃_le_linear`**, `radialRayKhat₃_bounded_near_zero`),
  **`radialRayKhat₃_eq_zero`** (the full trichotomy driver — sInf/sSup zeros
  with closure-image continuity, IVT midfields, metric ball extensions,
  `exists_Ioo_linear_bound_of_hasDerivAt_zero` at interior crossings, the
  `r = 0` universal edge), **`radialRayV₃_eq_zero`** (dense-zero gluing via
  `tendsto_nhds_unique_of_frequently_eq` — no interior-topology bookkeeping),
  and **`radialRayZ₃_proportional`** (`Z̃_p = c·Z̃_q` on `(0,∞)`, ratio-MVT
  with `C = 0`).  Gotchas: `Ioo_mem_nhdsGT hab : Ioo a b ∈ 𝓝[>] a`;
  `Convex.norm_image_sub_le_of_norm_hasDerivWithin_le` (Convex-prefixed);
  `integral_id` is top-level (not `intervalIntegral.`); `mono_left
  nhdsWithin_le_nhds` needs a type-ascribed `have`; `Ioi_mem_nhds hr` IS the
  eventually `0 < t`.
- 2026-07-16 (**DISCOVERY — the cheap invariance route**): the remaining
  `Z_p(x) = Z̃_p(‖x‖)` does NOT need polar coordinates or a 3-d CoV.
  **θ-flow argument**: for `w(θ,u,φ) = cosθ·u + sinθ·√(1−u²)cosφ` (the inner
  product of the chart point with a tilted axis), the rotation generator
  about `e₃` in chart coordinates, `X = (−√(1−u²)cosφ, −u·sinφ/√(1−u²))`, is
  **divergence-free** (`∂ᵤX_u + ∂φX_φ = 0`) and satisfies
  `∂θ w = −X·∇_{(u,φ)} w`.  Hence for smooth `F`,
  `I(θ) := ∫∫ F(w) dchartBase` has `I'(θ) = −∫∫ X·∇(F∘w) = 0` by TWO
  elementary 1-d IBPs: the `u`-boundary vanishes because
  `X_u(±1) = ∓0·cosφ = 0`, the `φ`-boundary because `sin(±π) = 0`.  So
  `I(θ) = I(0)`, i.e. the tilted-axis zonal average equals the polar one; the
  azimuth is handled by φ-periodicity.  Apply with
  `F(w) = exp(−√(‖x‖²+s²−2s‖x‖w)/τ)` (smooth for `s ≠ ‖x‖`); the collision
  shells are handled by: atoms of `ν` are countable ⟹ the identity holds for
  a dense set of `‖x‖` ⟹ everywhere by continuity of both sides in `x`.
  Then `c = 1` via the 3-d Tonelli mass identity (`∫Z_p = κ = ∫Z_q` with
  `κ = ∫e^{−‖x‖/τ}dx`, using `laplaceEuclideanFourierBase_integrable` from
  L4), and the headline concludes with
  `laplaceKernelNormalizer_injective_euclideanSpace…`/L4.
- 2026-07-15 (pre-implementation refinement): two further simplifications.
  (a) **The C¹ layer is y-level, not per-shell**: `Z̃', C̃' = D̃/τ, D̃'`
  all follow from dominated differentiation of the integrals over
  `radialMixture₃ μ̃` itself, with GLOBAL constant dominators
  (`|∂_r e^{-d/τ}| ≤ 1/τ`, `|∂_r(X e^{-d/τ})| ≤ 1 + sup(a·e^{-a/τ})/τ`, etc.)
  — differentiability holds for all `r > 0` at every `y` off the positive
  axis, and the off-origin axis `{u = ±1, s > 0}` is chartBase-null while
  `y = 0` has `d(r) = r`, smooth.  No per-shell differentiation, no
  Lipschitz-pairing dominators needed.  (b) **Only ONE per-shell identity
  is needed: T₃** (`∫(ρ²/d)w̄ = (2τ/r)∫ t·w̄` per shell).  Given T₃-mixture
  and the y-level formula `D̃' = -Z̃ + (1/τ)∫(X²/d)w̄ dμ`, the closure (F4)
  is y-level algebra (`d = X²/d + ρ²/d` a.e.), so the closed forms of
  `C̄, B̄, Q̄` are never needed.  Closed forms needed: `Z̄` (mass identity
  only) and the two T₃ kernels (`ρ²/d` and `t`).  T₃'s per-shell proof:
  the reverse `d`-substitution `u(z) = (r²+s²-z²)/(2rs)` (POLYNOMIAL — no
  √-singularity even at `r = s`, avoiding all endpoint case splits) via
  `intervalIntegral.integral_comp_mul_deriv'`, then primitives
  `Pₖ' = zᵏe^{-z/τ}` (k ≤ 4) and per-exponential-atom `ring`/
  `linear_combination`.  `ρ²(z) = [(z²-(r-s)²)((r+s)²-z²)]/(4r²)` vanishes
  at both endpoints.  The u-IBP route to T₃ is abandoned (√-singularity).
- 2026-07-15 (S-layer, drift component — **committed after green build**):
  `shellD` is implemented in `LaplaceRadialShell3.lean`, together with the
  public coordinate bound for `laplaceWeightedDisplacement`, the chart collapse
  `(laplaceWeightedDisplacement τ (rayProbe r) (s • sphereChart u φ)) 0 =
  exp (-(1/τ)·shellDist r s u)·(s·u-r)`, and the ray drift-coordinate mixture
  theorem `laplaceWeightedDisplacement_coord_radialMixture₃`.  This completes
  the three basic ray object formulas `Z̃`, `C̃`, and `D̃` as `ν`-mixtures of
  per-shell zonal averages.  Remaining S-layer: T₃ per-shell identity and the
  ray-mass identity.
- 2026-07-15 (S-layer, T₃ support infrastructure — **committed after
  `--wfail` build**): added the T₃ vocabulary in `LaplaceRadialShell3.lean`:
  `shellAxial`, `shellRhoSq`, `shellT`, `shellRhoSqOverDist`, the chart
  coordinate simp lemma, nonnegativity of `ρ²`, the algebraic cylindrical
  decomposition `shellDist² = shellAxial² + shellRhoSq`, and the useful
  relation `shellD = shellT - r·shellZ`.  Remaining S-layer hard analysis:
  the actual T₃ integral identity via the reverse `d`-substitution and the
  ray-mass identity.
- 2026-07-15 (S-layer, T₃ polynomial core — **green focused build**): added the
  reverse-distance polynomial numerator `shellRhoPoly`, its endpoint vanishing,
  derivative `P'(z)=4z(r²+s²-z²)`, the product derivative for
  `P(z)·exp(-(1/τ)z)`, and the endpoint-zero FTC identity.  The rearranged
  polynomial identity is now certified:
  `∫ P(z)e^{-z/τ} dz = 4τ ∫ z(r²+s²-z²)e^{-z/τ} dz` over
  `[|r-s|, r+s]`.  Remaining T₃ work: push this through the reverse
  `d`-substitution to identify the two original zonal shell integrals
  `shellRhoSqOverDist` and `shellT`; then prove the ray-mass identity.
- 2026-07-15 (S-layer, reverse-distance substitution algebra — **green focused
  build**): added `shellDistSubst r s z = (r²+s²-z²)/(2rs)`, its derivative
  `u'(z)=-z/(rs)`, endpoint values `u(|r-s|)=1`, `u(r+s)=-1`, the exact
  distance recovery `shellDist r s (u(z)) = z` for `z ≥ 0`, and the algebraic
  pullbacks for `ρ²` and `s·u(z)`.  Remaining T₃ work is now specifically the
  measure/interval change-of-variables layer: convert the `Ioc(-1,1)` set
  integrals defining `shellT` and `shellRhoSqOverDist` into interval integrals,
  apply `intervalIntegral.integral_comp_mul_deriv'` to `shellDistSubst`, and
  finish with `shellRhoPoly_integral_identity`.
- 2026-07-15 (S-layer, set-to-interval bridge — **green focused build**): added
  the generic rewrite `integral_Ioc_neg_one_one_eq_interval` and shell-specific
  interval forms `shellT_eq_intervalIntegral` and
  `shellRhoSqOverDist_eq_intervalIntegral`.  The remaining T₃ proof can now
  start from interval integrals on `[-1,1]` and apply the already-certified
  reverse-distance substitution algebra directly.
- 2026-07-15 (S-layer, substitution bridge for `shellT` — **green focused
  build**): added the generic change-of-variables lemmas
  `integral_comp_shellDistSubst_mul_deriv` and
  `integral_comp_shellDistSubst_mul_pos`, then applied them to `shellT`.
  Certified formulas:
  `shellT_eq_distance_intervalIntegral` and the factored
  `shellT_eq_polynomial_distance_integral =
  (4r²s)⁻¹ ∫ z(r²+s²-z²)e^{-z/τ} dz`.  Remaining T₃ proof: prove the analogous
  `shellRhoSqOverDist` z-form.  This is the one subtle endpoint-singularity
  step: for `r=s`, the `ρ²/d` integrand has a total-division value at the
  endpoint collision, so use the weaker
  `intervalIntegral.integral_comp_mul_deriv'''` route (continuous on the open
  image + integrable on the closed image), not the global-continuity wrapper.
- 2026-07-15 (S-layer, non-collision `ρ²/d` and T₃ — **green focused build**):
  added image-continuity substitution wrappers and proved the `ρ²/d` z-form
  when `r ≠ s`, then combined it with `shellRhoPoly_integral_identity` and
  `shellT_eq_polynomial_distance_integral` to certify
  `shellRhoSqOverDist τ r s = (2τ/r)·shellT τ r s` for `τ,r,s>0` and `r≠s`.
  The remaining T₃ gap is now exactly the removable collision case `r=s>0`.
- 2026-07-15 (S-layer, full positive-radius T₃ — **green focused build**):
  closed the removable collision case `r=s>0`.  Added the continuous regularized
  collision integrand, proved it agrees with the total-division integrand
  a.e. on `[-1,1]`, pushed it through the reverse-distance substitution, and
  derived the collision polynomial formula.  The final wrapper
  `shellRhoSqOverDist_eq_two_tau_div_r_mul_shellT` now proves T₃ for all
  `τ,r,s>0`.  Remaining S-layer item: the ray-mass identity
  `∫₀∞ r² shellZ τ r s dr = 2τ³`.
- 2026-07-15 (R-layer start — **green focused build**): created
  `LaplaceRadialRay3.lean`.  It packages the shell mixtures as ray-profile
  functions `radialRayZ₃`, `radialRayC₃`, `radialRayD₃`, `radialRayT₃`, and
  `radialRayRhoSqOverDist₃`, with bridge lemmas back to the kernel normalizer,
  companion normalizer, and first-coordinate drift numerator.  No C¹/system
  claims are made yet.  Next R-layer work after the ray-mass identity: add
  integrability hypotheses for algebraic mixture identities and then the
  dominated-differentiation layer.

**(F0) Scope decision.**  v1 targets `E = EuclideanSpace ℝ (Fin 3)` exactly —
the physically canonical dimension and the cleanest (`n = 3` makes the zonal
density *uniform*, Archimedes).  General `n ≥ 3` (v2) follows the same skeleton
with Gegenbauer densities `(1-u²)^{(n-3)/2}` and the Matérn ladder shifted; no
step below is `n = 3`-specific in *structure*, only in the constants and in the
closed forms being elementary.

**(F1) Radial measures via an explicit chart — no `toSphere`, no `bind`.**
Define `sphereChart (u φ : ℝ) : EuclideanSpace ℝ (Fin 3) :=
!₂[u, √(1-u²)·cos φ, √(1-u²)·sin φ]` and

```
radialMixture₃ (μ̃ : Measure ℝ) : Measure (EuclideanSpace ℝ (Fin 3)) :=
  Measure.map (fun z : ℝ × (ℝ × ℝ) => z.1 • sphereChart z.2.1 z.2.2)
    (μ̃.prod chartBase),
chartBase := (4π)⁻¹ • ((volume.restrict (Ioc (-1) 1)).prod
                        (volume.restrict (Ioc (-π) π)))
```

with hypotheses `[IsProbabilityMeasure μ̃]` and `hsupp : μ̃ (Iio 0) = 0`
(measures on ℝ, not ℝ≥0, to keep 1-d calculus painless; `s = 0` gives the
origin atom automatically since `0 • Φ = 0`).  ONE pushforward of ONE product
measure: no kernel-measurability, no `Measure.bind`.  Archimedes (`n = 3`
hat-box: the `u`-marginal of the sphere is uniform) holds *by definition*.
Integral collapse: `∫ f d(radialMixture₃ μ̃) = ∫∫∫ f(s•Φ(u,φ)) (4π)⁻¹ dφ du dμ̃`
by `integral_map` + `integral_prod`.  Key algebra: `‖s•Φ(u,φ)‖ = s` (s ≥ 0,
u ∈ [-1,1]), `⟪s•Φ, e₁⟫ = s·u`, and for the probe `x = r•e₁`:
`d² = ‖r•e₁ - s•Φ‖² = r² + s² - 2rsu` — no φ-dependence, so every ray
integrand collapses to a 1-d `u`-integral.

**(F2) The `d`-substitution and the Matérn-ladder closed forms.**  For
`r, s > 0`, substituting `u ↦ d = √(r²+s²-2rsu)` (`du = -(d/rs)·dd`, range
`d : r+s → |r-s|`) turns every per-shell kernel into an elementary integral.
With the ladder profiles

```
G₀(a) = e^{-a/τ},  G₁(a) = (a+τ)e^{-a/τ},  G₂(a) = (a²+3τa+3τ²)e^{-a/τ},
G₃(a) = (a³+3τa²+6τ²a+6τ³)e^{-a/τ}   (∫ aᵏe^{-a/τ}-antiderivative ladder:
∫ d·G₀ dd = -τG₁, ∫ (d²+τd)G₀-type dd = -τG₂, ∫ d³G₀ dd = -τG₃;
ladder ODE: G₁'' = G₁/τ² - (2/τ)G₀,  G₂'' = G₂/τ² - (4/τ)G₁),
```

the five per-shell kernels (`E₁ := G-at-|r-s|`, `E₂ := G-at-(r+s)` structure;
each is `(poly in r,s,τ)·e^{-|r-s|/τ} + (poly)·e^{-(r+s)/τ}`):

```
Z̄(r,s) := ⨍ e^{-d/τ}        = (τ/(2rs))·[G₁(|r-s|) - G₁(r+s)]
C̄(r,s) := ⨍ (d+τ)e^{-d/τ}   = (τ/(2rs))·[G₂(|r-s|) - G₂(r+s)]  (companion)
N̄(r,s) := ⨍ (1/d)e^{-d/τ}   = (τ/(2rs))·[G₀(|r-s|) - G₀(r+s)]  (Green; unused v1)
B̄(r,s) := ⨍ (t-r)e^{-d/τ}   = (τ/(4r²s))·[(s²-r²)(G₁(|r-s|)-G₁(r+s))
                                            - (G₃(|r-s|)-G₃(r+s))]  (drift)
Q̄(r,s) := ⨍ (X²/d)e^{-d/τ}  = (1/(8r³s))·∫_{|r-s|}^{r+s}(s²-r²-d²)² e^{-d/τ} dd
```

(t = ⟨y,e₁⟩, X = t - r = (s²-r²-d²)/(2r) after substitution; ρ² = s²-t²;
`s = 0` degenerates to `Z̄ = e^{-r/τ}` etc.)  **Crucial regularity bonus:
`G₁(|·|) and G₂(|·|) are C² on ℝ** (`G₁(|a|) = τ + 0·|a| - a²/(2τ) + O(|a|³)` —
the kink cancels; `G₂` likewise) — so `Z̄, C̄` are C² in `r` *everywhere
including r = s*, for every shell.  All (R6) worries (`∫dμ/d`, shells through
the probe, n=2-style log divergences) are gone at the source.  Uniform
dominators for differentiation under `∫dμ̃`: the Lipschitz-pairing bound
`(1/(2s))|G(|r-s|) - G(r+s)| ≤ Lip(G)·min(r,s)/s ≤ Lip(G)` handles `s → 0`.

**(F3) Ray objects and the C¹ layer (all that's needed — no C² layer!).**
`Z̃_μ(r) = ∫ Z̄(r,s) dμ̃`, `C̃_μ(r) = ∫ C̄ dμ̃`, `D̃_μ(r) = ∫ B̄ dμ̃`
(= e₁-component of the drift numerator `∫ e^{-d/τ}(y - re₁) dμ(y)`; tangential
components vanish by the trivial φ-integrals).  `Z̃ > 0` everywhere.
Differentiation under the μ̃-integral (dominated, Lipschitz-pairing bounds)
gives `Z̃, C̃, D̃ ∈ C¹((0,∞))` with `C̃' = D̃/τ` (per-shell: `∂_r C̄ = B̄/τ`,
the ray restriction of L0's `∇ψ = D`).  The zero-drift reduction:
`ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q` at `x = r•e₁` gives
`D̃_p Z̃_q = D̃_q Z̃_p`, so `m̃ := D̃_p/Z̃_p = D̃_q/Z̃_q` is the common tilted
displacement, `m̃ ∈ C¹` by quotient rule.  **The system never needs `Z̃''`,
`w̃'`, or any second derivative** — see (F5).

**(F4) The closure identity (the radial analogue of 1-d `C = τD' + 2τZ`).**
Hand-derived and double-checked:

```
(CLOSURE, n=3)     C̃_μ = τ·D̃_μ' + 4τ·Z̃_μ + (2τ/r)·D̃_μ      on (0,∞)
(general n:        C̃  = τD̃' + (n+1)τZ̃ + (n-1)τD̃/r ;  n=1 is the proven 1-d identity)
```

Lean route: per-shell algebraic identity `C̄ = τ·∂_r B̄ + 4τ·Z̄ + (2τ/r)·B̄`
— pure polynomial–exponential algebra in the two atoms `e^{-|r-s|/τ}`,
`e^{-(r+s)/τ}` (case split `r ≶ s`, then `ring`-normalizable per atom) — then
integrate `dμ̃`.  **The tangential IBP (T) of (R2)/(R8) is therefore NEVER
formalized as an IBP: it dissolves into this closed-form identity.**  Same for
the (T)-corollaries used by the sign layer (F6).

**(F5) The K̂-system (verified derivation).**  Set `w̃ := Z̃_p'Z̃_q - Z̃_q'Z̃_p`,
`v := r²·w̃`, `K := C̃_pZ̃_q - C̃_qZ̃_p`, `K̂ := r²·K`.  Under zero drift, using
only (F3)+(F4) and algebra:

```
K = τ·m̃·w̃            (companion defect = τ·m̃·Wronskian; so K̂ = τ·m̃·v)
K' = C̃_pZ̃_q' - C̃_qZ̃_p' = -τ·(m̃' + 4 + 2m̃/r)·w̃
K̂' = -τ·(m̃' + 4)·v     (the r²-weight absorbs the geometric term — (R3) verified)
```

On `{m̃ ≠ 0}`: `v = K̂/(τm̃)`, so `K̂' = -[(m̃'+4)/m̃]·K̂` — the Abel ODE with
coefficient exactly of the 1-d wrappers' hard-coded shape `c = 2·μ̂/m̃` for
**`μ̂ := (m̃'+4)/2`** (1-d had `(m'+2)/2`).  The LaplaceACPropagation layer
(whose core lemmas are stated for a *general* coefficient `c`, with `2μ/m`
only in thin wrappers) applies **unchanged** — no symbolic-pair edit needed.

**(F6) Sign layer — better than (R5): the open region shrinks to `m̃ > r`.**
Pointwise formulas (from (F3) quotient rule + the per-shell identity
`∂_r B̄ = -Z̄ + Q̄/τ`, again pure algebra):
`m̃' + 1 = (1/τ)·Cov_w(X, X/d)` where `Cov_w` is the `e^{-d/τ}·μ`-tilted
covariance at probe `r`.  Then:

- **Zero case (free):** `m̃(r) = 0 ⟹ m̃' + 1 = E_w[X²/d]/τ ≥ 0`, so `μ̂ ≥ 3/2`.
- **`m̃ ≤ r` case (PROVED, unified — subsumes all of `m̃ ≤ 0`):**
  Cauchy–Schwarz in `L²(w̄dμ)` gives `|m̃| ≤ E|X| ≤ √(E[X²/d]·E[d])`;
  the (T)-corollary `E_w[d] = E_w[X²/d] + 2τ·c(r)/r` (`c := E_w[t] = r + m̃`;
  per-shell algebra + `d = X²/d + ρ²/d` a.e.) with `m̃ ≤ r ⟹ c ≤ 2r` yields
  `|m̃| ≤ Q + 2τ`, hence `Cov ≥ Q - m̃·|E[X/d]| ≥ -2τ`, i.e. **`m̃' ≥ -3` and
  `μ̂ ≥ 1/2` pointwise on `{m̃ ≤ r}`** — full RSI strength with the same margin
  as 1-d.  (`|X/d| ≤ 1`; `|X| = √(X²/d)·√d` off the μ-null axis.)
- **`m̃ > r` (OPEN — the far-tilt regime, `E_w[t] > 2r`):** v1 ships the named
  hypothesis **`RadialSlack₃ : ∀ r > 0, r < m̃ r → -3 ≤ m̃' r`** (μ̂ ≥ 1/2
  there).  Within-shell covariance is ≥ 0 for every shell with `s > r`
  (`X` and `X/d` co-monotone in `u` iff `t < s²/r`, always true for `s > r`),
  so a violation needs cross-shell anticorrelation against the `2τ` buffer —
  numerics spec: near+far shell mixtures, scan `Q + 2τ - m̃E[X/d]` and `m̃'+3`
  directly, focused on configurations with `m̃ > r`.
- `radialCentroid_nonneg (c ≥ 0)` is NOT on the critical path (dropped).

**(F7) Trichotomy over `(0,∞)` — exact port map of :882–:1010.**  For
`x₀ ∈ (0,∞)`:
- `m̃(x₀) = 0`: `K̂(x₀) = τ·m̃·v = 0` directly.
- `m̃(x₀) < 0`: if a zero of `m̃` exists in `[x₀,∞)`, take the infimum zero β:
  on `(l₁, β)` use `abel_left_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_lower`
  (edge at β; near β, `m̃` is small < r so μ̂ ≥ 1/2 is in the PROVED region;
  `m̃ ≥ -L(β-t)` from local m̃'-boundedness via MVT).  Else `m̃ < 0` on the whole
  ray: `abel_right_outer_zero_of_muDeriv_nonneg_of_m_neg` with
  `K̂ → 0 at ∞` (F8).
- `m̃(x₀) > 0`: if a zero exists in `(0, x₀]`, take the supremum zero α: on
  `(α, b)` (∀b) use `abel_right_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_upper`
  (edge at α; near α proved-region μ̂ ≥ 1/2; `m̃ ≤ L(t-α)` via MVT).  Else
  `m̃ > 0` on all of `(0, x₀]`: **r = 0 is the universal edge**: `m̃(0⁺) = 0`
  with `m̃ ≤ L·r` near 0 (local m̃'-bound; no continuity of m̃' needed —
  pointwise derivative bounds + MVT), μ̂ ≥ 1/2 near 0 (proved region if
  `m̃ ≤ r`, RadialSlack₃ if `r < m̃ ≤ Lr`), so `c ≥ 1/(L·r)` — the same
  log-singularity lemma fires with `a = 0`.  NO left-tail lemma exists or is
  needed (`(0,∞)` has no `-∞` end; `m̃ > 0`-rays terminate at the r=0 edge).
- Needed K̂ inputs: continuity on `(0,∞)` ✓(C¹); `HasDerivAt K̂ (-τ(m̃'+4)v)` ✓;
  boundedness near each edge ✓; `|K̂| ≤ C·r²` near 0 (`|C̃|,|Z̃| ≤ const`,
  `|D̃| ≤ τ/e`).

**(F8) Boundary at ∞.**  `|K̂| ≤ r²(C̃_pZ̃_q + C̃_qZ̃_p)` and each factor
splits `s ≶ r/2`: `Z̃_μ(r) ≤ G₁-scale·e^{-r/(2τ)} + μ̃([r/2,∞))`.  Under
**first moments** (`∫ s dμ̃ < ∞`, both measures — the v1 hypothesis, =
(n-1)/2-moments at n=3): `r·μ̃([r,∞)) = r∫1_{s≥r} ≤ ∫s·1_{s≥r} → 0` by
dominated convergence — genuinely → 0, not just bounded — so
`r²·tail_p(r/2)·tail_q(r/2) → 0` and `K̂ → 0` at ∞ **along the full filter**
(no subsequence subtleties).

**(F9) Endgame — ray-only up to the last step, then L4.**
1. `K̂ ≡ 0 ⟹ v ≡ 0`: on `{m̃≠0}`, `v = K̂/(τm̃) = 0`; on `int{m̃=0}`,
   `0 = K̂' = -τ(m̃'+4)v = -4τv`; the union is dense and `v` continuous —
   reuse `continuous_eq_zero_of_dense_zeroSet` (LaplaceACPropagation:1343).
2. `w̃ = v/r² ≡ 0 ⟹ (Z̃_p/Z̃_q)' ≡ 0 ⟹ Z̃_p = c·Z̃_q` on `(0,∞)`.
3. **`c = 1` by the ray mass identity** (new, s-independent per shell —
   verified): `∫₀^∞ r²·Z̄(r,s) dr = 2τ³` for every `s ≥ 0` (including s = 0),
   via `∫₀^∞ r[G₁(|r-s|)-G₁(r+s)]dr = 2s∫₀^∞G₁ = 4sτ²`; Tonelli ⟹
   `∫₀^∞ r²Z̃_μ = 2τ³` for EVERY radial-mixture probability μ — so `c = 1`.
4. Rotation invariance ⟹ `Z_p = Z_q` on all of E: prove
   `kernelNormalizer (laplaceKernel τ) (radialMixture₃ μ̃) x = Z̃(‖x‖)` from
   O(3)-invariance of `radialMixture₃`, which follows from the **polar
   formula** `∫_{ℝ³} f = ∫₀^∞ 3·vol(B₁)·s²·(⨍_{shell s} f) ds`-shape proved by
   Fubini + `polarCoord` (Mathlib has `integral_comp_polarCoord`-form) + one
   2-d change of variables `(R,u) ↦ (Ru, R√(1-u²))` (Jacobian `R/√(1-u²)`;
   at n=3 the √ cancels exactly — Archimedes), THEN invariance of volume under
   linear isometries transfers to the chart measure (for any isometry R fixing
   0 and any radial-profile test slot).  Point-transport: Householder
   reflection through the bisector hyperplane of `x` and `‖x‖e₁`
   (`reflection` in Mathlib's InnerProductSpace/Projection) maps `‖x‖e₁ ↦ x`.
5. Feed L4: `laplaceKernelNormalizer_injective_euclideanSpace_of_fourier_ne_zero`
   / `laplaceSmoothingInjective_euclideanSpace` (ι := Fin 3) concludes
   **`p = q`**.
   Fallback if the polar/invariance CoV fights Lean: the half-line Green
   uniqueness on `(1-τ²∂²)(rZ̃)`-data (kernel `e^{-|r-s|/τ} - e^{-(r+s)/τ}`,
   Wronskian ≡ 1) or a 1-d charFun argument on the odd extension — both
   ray-only, no invariance needed.

**(F10) What is NOT built (dissolved layers).**  No `Measure.toSphere`, no
zonal-density construction, no `Measure.bind`, no tangential-IBP lemma, no
C²/Δ layer, no `∫dμ/d` local-uniform estimates, no second-derivative
formulas, no `radialCentroid_nonneg`, no signed measures.  The only genuinely
new analytic ingredients are: one interval `d`-CoV, dominated C¹
differentiation with Lipschitz-pairing dominators, one L² Cauchy–Schwarz on
`w̄dμ`, per-shell polynomial–exponential `ring` identities, and (for the
endgame only) one 2-d CoV + linear-isometry volume invariance.

**(F11) File plan (v1).**
1. `LaplaceRadialShell3.lean` — chart, `radialMixture₃`, integral collapse,
   `d`-CoV, the five closed forms + s=0 forms, per-shell identities
   (closure (F4), `∂_rB̄ = -Z̄ + Q̄/τ`, (T)-corollary), Lipschitz-pairing
   dominator lemmas, ray mass identity per shell.
2. `LaplaceRadialRay3.lean` — `Z̃/C̃/D̃/m̃`, positivity, C¹ layer, mixture
   closure, zero-drift ray reduction, `m̃'`-formula, sign layer
   (zero case, `m̃ ≤ r` C–S case, `RadialSlack₃`), local m̃'-bounds/MVT
   Lipschitz lemmas.
3. `LaplaceRadialSystem3.lean` — `w̃, v, K, K̂`, the (F5) identities,
   `|K̂| ≤ Cr²`, `K̂ → 0` at ∞ (first moments), trichotomy (F7) ⟹ `K̂ ≡ 0`,
   endgame steps 1–3 (`Z̃_p = Z̃_q`).
4. `LaplaceRadialInvariance3.lean` — polar formula, O(3)-invariance,
   `Z_p(x) = Z̃_p(‖x‖)`, Householder transport.
5. `LaplaceRadialConverse3.lean` — headline
   `laplaceZeroDrift_identifies_of_radialMixture₃` (hypotheses:
   `IsProbabilityMeasure μ̃_p/q`, supports in `[0,∞)`, first moments,
   `RadialSlack₃` for both, `ZeroDrift`) + audit entries.  If numerics
   confirm/prove `m̃ > r` later, drop `RadialSlack₃` for the unconditional
   form.

**Risks & fallbacks.**  (i) `RadialSlack₃` open — numerics decide honesty
(conjectured-true hypothesis vs necessary); theorem is real either way.
(ii) Endgame invariance CoV — bounded, with two ray-only fallbacks (F9.5).
(iii) Mathlib L²-C-S form — worst case, elementary `(∫fg)² ≤ ∫f²∫g²` by
expanding `∫∫(f(x)g(y)-f(y)g(x))² ≥ 0`.  (iv) The `!₂[..]` notation /
EuclideanSpace-literal API — cosmetic only.

---

## 6. Recon inventory (verified this pass)

| asset | location | note |
|---|---|---|
| `laplaceKernel` norm-generic | Paperaxioms.lean:250 | both ℓ¹ (on `PiLp 1`) and ℓ² (on `EuclideanSpace`) are instances |
| `meanShift`/`meanShiftDrift`/`ZeroDrift` E-generic | Paperaxioms.lean:83–213 | `Distribution = Measure` (:44) |
| 1-d unconditional converse | LaplaceUnconditionalConverse.lean:1013 | `laplaceZeroDrift_identifies`, arbitrary probability measures |
| 1-d smoothing injectivity, finite measures | LaplaceInjectivity.lean:408 | `laplaceKernelNormalizer_injective`; predicate `LaplaceSmoothingInjective` :439 |
| generic conv/charFun cancellation | GaussianConvolutionInjectivity.lean:32 | reusable for radial-smoothing injectivity in D2 |
| n-d radial far-field machinery | LaplacianGaussianConverse.lean:30–315 | distribution-generic with exponential-moment domination |
| n-d Laplace ⊘ Gaussian-target converse | LaplacianGaussianConverse.lean:886 | closed; arbitrary-target is the open part |
| Gaussian n-d arbitrary converse | GaussianScoreRecovery.lean:295 / GaussianArbitraryConverse.lean | headline parity pattern to mirror |
| π-system extension | Mathlib `ext_of_generate_finite`, `isPiSystem_pi`, `pi_eq_generateFrom` | Typeclasses/Finite.lean:448; Constructions/Pi.lean:263,754 |
| density/map integral rewrites | Mathlib Bochner/ContinuousLinearMap.lean:250,295 | plus `ContinuousLinearMap.integral_comp_comm`, `ContinuousLinearMap.proj` |
| `PiLp` measurable structure | Mathlib Analysis/Normed/Lp/MeasurableSpace.lean:26 | `WithLp.measurableSpace` comap; Borel instances same file |
| parametric-integral derivative | Mathlib Analysis/Calculus/ParametricIntegral.lean:165 | `hasFDerivAt_integral_of_dominated_loc_of_lip` — powers L0's `D = ∇ψ` |
| Gaussian Fourier transform | Mathlib Analysis/SpecialFunctions/Gaussian/FourierTransform.lean | feeds L4's subordination positivity |
| ball averages / measures | Mathlib `MeasureTheory.average`, `measure_ball_pos` (OpenPos.lean:212) | feeds L3's cone extraction (solid balls, no sphere measure needed) |
| generic charFun cancellation hypotheses | GaussianConvolutionInjectivity.lean:32 | takes any finite `p q ν` on inner-product `E`; the file's :12–14 header confirms the Fourier-side factoring deliberately avoids density-of-convolution API — L4 mirrors it verbatim |

Hand-derived and cross-checked this pass (not yet in Lean): the §2 reduction
(each step re-verified against the exact statements above); (♦) checked against
`p = δ₀, n = 1` and against the repo's 1-d companion identities; `g - τ²Δg =
(n+1)τ²k` and `g' = -re^{-r/τ}` checked symbolically; (★) checked to reduce to
the proven 1-d Abel ODE at `n = 1`.

---

## 7. Implementation addendum: L3 atom-alignment gate

Added 2026-07-14.  The algebraic gate for L3 is implemented axiom-free in
`LaplaceAtomAlignment.lean` and wired into the root module.

Implemented names:

- `ballAverage`
- `laplaceAtomConeScale`
- `laplaceAtomConeCoeff`
- `atomMassReal`
- `LaplaceAtomConeProductData`
- `laplaceNormalizerDisplacementProduct_eq_of_zeroDrift`
- `laplaceZeroDrift_atomAlignment_of_coneProductData`
- `laplaceZeroDrift_atomMassRatio_of_coneProductData`
- `laplaceZeroDrift_atomMass_zero_iff_of_coneProductData`

Interpretation: if the planned small-ball cone extraction proves the two product
limits packaged by `LaplaceAtomConeProductData`, then zero drift immediately
forces the atom-alignment identity
`q({a}) * D_p(a) = p({a}) * D_q(a)` in vector-smul form, plus the nonzero-drift
atom-mass ratio and atom-presence equivalence.

Remaining L3 work: prove the analytic cone-extraction package for arbitrary
probability measures in dimension `n >= 2`:

1. normalizer cone coefficient equals atom mass;
2. displacement numerator cone coefficient is zero;
3. product rule for `Z_nu * D_mu`.

No axiom, constant, or opaque theorem was introduced by the L3 gate.
