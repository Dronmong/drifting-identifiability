# Laplace-kernel converse for arbitrary targets: attack plan and research record

**Status:** Stages 1–2 completed and audited (2026-07-09): the
companion-kernel score identity, no-moment regularity, cross-gradient
reformulation and tilt bridge (`LaplaceCompanion.lean`); real-line smoothing
injectivity and Dirac rigidity (`LaplaceInjectivity.lean`).  Stage 3 first
increment landed 2026-07-10 (`LaplaceWronskian.lean`): the 1-d elliptic
structure is CLASSICAL (`D_p′ = L_p/τ - 2Z_p`, hence `τ²L_p″ = L_p - 2τZ_p`,
no distributions), and the **alignment reduction** — zero drift plus
`K := L_p·Z_q - L_q·Z_p ≡ 0` forces `p = q` — is machine-checked, reducing
the open 1-d converse to the single scalar identity `K ≡ 0`.  All axiom-free.
The full arbitrary-target Laplace converse remains open.

## The problem

The single remaining cell of the rebuttal's kernel × target matrix, conceded
open by the authors ("the general converse for arbitrary fields remains
open"), sharpened by this project to its honest form:

```text
For the paper's practical kernel  kτ(x,y) = exp (-‖x-y‖/τ),  τ > 0:
does pointwise zero raw mean-shift drift between two arbitrary probability
measures force p = q?

ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q  →  p = q ?
```

Known context (all machine-checked in this repo):

- Gaussian kernel + arbitrary targets: TRUE (`gaussianMeanShiftDrift_
  identifiesAtZero`, axiom-free). Mechanism: the Gaussian score identity —
  the mean shift is the gradient of `log` of the smoothed density.
- Laplace kernel + Gaussian targets: TRUE (`laplaceGaussianMeanShiftDrift_
  identifiesAtZero`, axiom-free). Mechanism: radial exponential tilts recover
  the finitely many parameters.
- Arbitrary kernels: FALSE (flat/band-limited counterexamples; recorded).
- The Laplace kernel is spectrally fine: its Fourier transform (a Poisson/
  Cauchy-type profile `∝ τ^d (1+τ²‖ξ‖²)^{-(d+1)/2}`) is everywhere positive,
  so `Z_p = Z_q → p = q` (convolution injectivity) is NOT the obstruction.
- The obstruction is normalization: zero drift only matches the RATIOS
  `A_p/Z_p = A_q/Z_q` where `Z_p = kτ * p`, `A_p = kτ * (y·p)`, and for the
  Laplace kernel the mean shift is NOT the score of `Z_p`
  (`∇ₓkτ` carries weight `kτ/‖x-y‖`, not `kτ`).

## The new structural handle (Stage 1): the companion-kernel score identity

**Finding.** Although the Laplace mean shift is not the score of the Laplace
smoothing, it IS an exact weighted gradient — of a *different* radial
smoothing. Define the **companion kernel**

```text
ℓτ(x,y) := (τ + ‖x-y‖) · exp (-‖x-y‖/τ) = (τ + ‖x-y‖) · kτ(x,y),
L_p(x)  := ∫ ℓτ(x,y) dp(y)      (the companion normalizer).
```

Then, pointwise in the kernel (verified numerically to 1e-11, gradient
identity including at `x = y` where both sides vanish):

```text
∇ₓ ℓτ(x,y) = -(1/τ) (x-y) kτ(x,y),
```

so for EVERY probability measure `p` (no moment assumptions — see below):

```text
∇L_p(x) = (1/τ) ∫ kτ(x,y) (y-x) dp(y) = (1/τ) D_p(x),
meanShift (laplaceKernel τ) p x = (τ / Z_p(x)) ∇L_p(x).
```

Hence the **cross-gradient reformulation** of the open problem:

```text
ZeroDrift ⟺ ∀x,  Z_q(x) ∇L_p(x) = Z_p(x) ∇L_q(x).
```

Remarks.

- This is the exact analogue of the Gaussian score identity, with the twist
  that the gradient potential (`L_p`) and the normalizer (`Z_p`) are DIFFERENT
  smoothings of `p`. For the Gaussian they coincide (up to constants) — that
  is precisely the miracle the Gaussian proof exploited. The whole difficulty
  of the open problem is now isolated in the mismatch `ℓτ ≠ c·kτ`.
- Where the derivative comes from: `(τ+s)e^{-s/τ}` has derivative
  `-(s/τ)e^{-s/τ}`, which vanishes at `s = 0`; equivalently
  `h(t) := (τ+√t)e^{-√t/τ}` is `C¹` on `[0,∞)` with `h'(t) = -e^{-√t/τ}/(2τ)`.
  So `x ↦ ℓτ(x,y)` is differentiable EVERYWHERE, including `x = y` (where the
  bare kernel `kτ` has its cone singularity). Elementary bound used at `x=y`:
  `0 ≤ τ - (τ+s)e^{-s/τ} ≤ s²/τ` (from `1+u ≤ e^u` twice).
- Subordination context (motivation only): `e^{-s/τ}` is a mixture of
  Gaussians over bandwidths; applying the Gaussian score identity under the
  mixture and integrating the bandwidth weight yields exactly
  `ℓτ/2` — the companion kernel is the Bessel `K_{3/2}` profile
  `½(1+s/τ)e^{-s/τ}` (scaled). The direct derivative proof above avoids
  subordination entirely and is what gets formalized.

**No-moment well-definedness (bonus finding).** The Laplace field needs NO
moment hypotheses: `‖kτ(x,y)•(y-x)‖ = s·e^{-s/τ} ≤ τ` is bounded, so `D_p`
(and the full `MeanShiftRegularAt` package, including the equation-11 product
integrability, bounded by `2τ·kτ`) holds for ARBITRARY probability measures at
every point. The raw Laplace field is globally defined on all of
`𝒫(E) × 𝒫(E) × E` — worth certifying on its own.

## What the asymptotics cannot do (recorded finding)

The radial-limit machinery (`LaplacianGaussianConverse.lean`, parts A–B,
distribution-generic) shows: for measures with the critical exponential moment
(`Integrable (exp(‖y‖/τ)(1+‖y‖))`), zero drift forces the exponential-tilt
centroids to agree in every unit direction — i.e. `∇ log M_p = ∇ log M_q` on
the SPHERE `‖a‖ = 1/τ` of tilt parameters (the tilt centroid is the tilted
mean, already proved generically). Two limitations, both structural:

1. Offset or curved escaping probes give nothing new: along `x₀ + r·u` the
   compensated weight limit is `e^{⟪u,y-x₀⟫/τ}` and the `x₀` factor cancels in
   the centroid ratio. All escaping probe families see only the sphere data.
2. A sphere is not a uniqueness set for analytic functions in `d ≥ 2` (e.g.
   `‖a‖²-1/τ²` vanishes on it), and in `d = 1` it is two points. So the sphere
   data determines finite-parameter families (Gaussians: `μ + Σu/τ`) but NOT
   general `p`. The interior information must come from finite probes — the
   asymptotic route is exhausted. This explains precisely why the authors'
   argument 2 works for Gaussian targets and cannot extend as-is.

## 1-d research notes (for later stages; verified numerically where marked)

In `d = 1` the Laplace kernel is the Green's function of an elliptic operator:

```text
(τ²∂² - 1) Z_p = -2τ·p     (distributionally; verified numerically),
(τ²∂² - 1) A_p = -2τ·(y·p).
```

Writing `m := meanShift` (common to both by zero drift), `Σ := Z_p + Z_q`,
`W := Z_p - Z_q`, eliminating `p ± q` yields TWO linear ODEs with the SAME
operator: `(τ²∂²-1)(mF) = -2τ²F'` for `F ∈ {Σ, W}`. Consequences worth
pursuing:

- If `W ∝ Σ` then integrating (`∫Z_p = ∫Z_q = 2τ`) forces `W ≡ 0`, hence
  `p = q` by 1-d convolution injectivity (the elliptic inversion above).
  Otherwise `Σ, W` form a fundamental system of a second-order operator whose
  leading coefficient is `m` — degenerating exactly at the zeros of the mean
  shift. The cross-Wronskian satisfies the exact measure identity
  `(Z_p W' - Z_p' W + ...)' `— specifically `J := Z_p'... ` in the clean form:
  `(Σ W' - Σ' W)' = (4/τ)(Z_p·dq - Z_q·dp)`. Global integration recovers only
  the symmetric identity `∬kτ dp dq = ∬kτ dq dp`; weighted test functions are
  the open direction.
- **Rigidity mini-theorem (constant conditional mean ⟹ Dirac):** if
  `C_p ≡ c` then `∫(y-c)kτ(x,y)dp(y) = 0` for all `x`, i.e. the signed measure
  `(y-c)·dp` has identically zero Laplace smoothing, hence is zero
  (convolution injectivity / elliptic inversion), hence `p = δ_c`. Clean
  target for a later Lean stage once smoothing-injectivity for the Laplace
  kernel is formalized.
- **Translation fields are impossible:** `m ≡ const ≠ 0` forces
  `p = (τ/m)Z_p'` a.c., and the resulting constant-coefficient ODE for `Z_p`
  has no positive solution decaying at both ends. (Sanity check on rigidity.)

## Staged plan

- **Stage 1 (completed, Lean, axiom-free, promoted):** new module
  `DriftingIdentifiability/LaplaceCompanion.lean` in the audited main track:
  1. `laplaceCompanionKernel τ x y := (τ + ‖x-y‖) * laplaceKernel τ x y`;
  2. pointwise gradient: `HasFDerivAt (fun x => ℓτ(x,y))
     (-(1/τ) kτ(x,y) ⟪x-y, ·⟫) x` — two cases (`x ≠ y` chain rule through the
     norm; `x = y` direct little-o with the `s²/τ` bound);
  3. no-moment integrability package: `D_p` integrable, equation-11
     `MeanShiftRegularAt (laplaceKernel τ) p q x` for ALL probability pairs
     and every `x`;
  4. differentiation under the integral (constant dominator `1/... ≤ 1`,
     mirroring `hasFDerivAt_gaussianKernelNormalizer`):
     `HasGradientAt (kernelNormalizer (laplaceCompanionKernel τ) p) ((1/τ) • D_p x) x`;
  5. the score-analogue identity
     `D_p x = kernelNormalizer (laplaceKernel τ) p x • meanShift (laplaceKernel τ) p x`
     and the headline cross-gradient reformulation
     `ZeroDrift ↔ ∀ x, Z_q x • ∇L_p x = Z_p x • ∇L_q x`;
  6. bridge to the tilt machinery: for measures with the critical exponential
     moment, `ZeroDrift → exponentialTiltCentroid p τ u = exponentialTiltCentroid q τ u`
     for every unit `u` (via the generic radial limit + uniqueness of limits).
  Registered in the default root and promoted axiom audit; full `Check.ps1`
  passes, and `#print axioms` on the promoted companion declarations reports
  only Lean foundations.
- **Stage 2 (completed for the real line, 2026-07-09): Laplace smoothing
  injectivity + Dirac rigidity.**  Implemented in
  `DriftingIdentifiability/LaplaceInjectivity.lean`:
  1. *1-d Fourier transform of the Laplace profile, by hand.*  Mathlib has no
     Poisson/Cauchy-kernel transform, but it has everything needed to compute
     it: split `ℝ = Iic 0 ∪ Ioi 0`; on each half-line the integrand is
     `exp (c± x)` with `c± = ∓1/τ - 2πit`, whose antiderivative
     `exp (c± x)/c±` decays (`‖exp (c± x)‖ = e^{∓x/τ}`), so
     `integral_Ioi_of_hasDerivAt_of_tendsto'` /
     `integral_Iic_of_hasDerivAt_of_tendsto'` evaluate both halves
     (integrability from `exp_neg_integrableOn_Ioi` + reflection).  Result:
     `𝓕(e^{-|·|/τ})(t) = (2/τ)/((1/τ)² + 4π²t²)` — real, positive, nowhere
     zero.
  2. *Injectivity on `ℝ`.*  Mirror the battle-tested skeleton of
     `GaussianConvolutionInjectivity.lean` verbatim with the Laplace profile
     (Fubini swap, `VectorFourier.fourierIntegral_comp_add_right`
     translation, cancel the nowhere-zero transform, `Measure.ext_of_charFun`)
     to get `laplaceKernelNormalizer_injective` for FINITE measures on `ℝ`:
     equal Laplace smoothings force equal measures.  General `d` needs the
     Bessel-type transform — recorded open.
  3. *Reusable injectivity interface.*  `LaplaceSmoothingInjective E τ`
     records the finite-measure smoothing-injectivity predicate without hiding
     any mean-shift conclusion; `laplaceSmoothingInjective_real` instantiates
     it on `ℝ`.
  4. *Signed-moment bridge.*  If
     `ZeroDrift (meanShiftDrift (laplaceKernel τ)) p (dirac c)`, then
     `∫ kτ(x,y)(y-c) dp = 0` for every probe `x`.  The positive and negative
     first-moment reweightings of `p` therefore have identical Laplace
     smoothings (`laplaceMomentParts_eq_of_zeroDrift_dirac_real`).
  5. *1-d Dirac rigidity:* `laplaceZeroDrift_dirac_identifies_real` proves
     that a probability law on `ℝ` with finite first moment and zero raw
     Laplace drift against `dirac c` must be exactly `dirac c`.  The proof uses
     smoothing injectivity to identify the positive/negative moment measures,
     their mutual singularity to make both vanish, and then the resulting
     `p`-a.e. identity `y = c`.
  Registered in the default root and promoted axiom audit; full `Check.ps1`
  passes, and `#print axioms` on the Stage-2 declarations reports only Lean
  foundations.  What remains open is the full arbitrary-target converse and
  higher-dimensional Laplace smoothing injectivity/Dirac rigidity (the
  Bessel-transform route or an equivalent theorem).
- **Stage 3 (in progress 2026-07-10): the 1-d ODE/Wronskian program.**  Two
  new findings (derived on paper this session, then formalized in
  `DriftingIdentifiability/LaplaceWronskian.lean`):

  1. *The elliptic structure is classical, not distributional.*  The 1-d
     displacement integrand `x ↦ kτ(x,y)·(y-x)` is `C¹` EVERYWHERE — the
     `sgn` singularity of `∂ₓkτ` is killed by the vanishing factor `(y-x)` —
     with derivative `(|x-y|/τ - 1)·kτ(x,y)`, uniformly bounded by `2`.
     Differentiating under the integral (constant dominator, the Stage-1
     pattern), the mean-shift numerator `D_p` is `C¹` with

     `D_p′ = M_p/τ - Z_p`,   where `M_p := ∫|x-y| kτ dp = L_p - τ Z_p`,

     so the companion normalizer is `C²` with the POINTWISE ode

     `τ² L_p″ = L_p - 2τ Z_p`    (equivalently `τ D_p′ = L_p - 2τ Z_p`).

     The plan's distributional identity `(τ²∂²-1)Z_p = -2τp` is thereby
     bypassed: everything the program needs lives at the level of classical
     derivatives of smoothings, which Lean handles with the existing
     machinery.  Bonus corollary: zero drift is equivalent to the Wronskian
     identity `Wr(L_p, L_q) = τ²·Wr(L_p′, L_q′)` — the "two solutions of one
     operator" structure, now classical.

  2. *The Wronskian/alignment reduction.*  Define the **companion alignment**

     `K(x) := L_p(x)·Z_q(x) - L_q(x)·Z_p(x)`.

     Under zero drift, `K ≡ 0` forces `p = q`, by a chain that is now fully
     certified on the line: zero drift gives `Z_q·D_p = Z_p·D_q` (Stage 1);
     combined with alignment and `Z_p > 0` this kills the Wronskian
     `L_p′L_q - L_pL_q′ ≡ 0`, so `L_p = c·L_q` (ratio has zero derivative);
     alignment again turns this into `Z_p = c·Z_q`; Stage-2 smoothing
     injectivity applied to `p` and `c•q` gives `p = c•q`, and total mass
     forces `c = 1`.  **The open 1-d converse is therefore reduced to the
     single scalar identity `K ≡ 0`** — one continuous function of one
     variable, with `K → 0` at both ends and (research note, distributional)
     `K″ = -(2/τ)(L_p dq - L_q dp)` as measures, so `K` is convex where
     `v·dp - u·dq ≥ 0` and concave where `≤ 0`; a maximum-principle argument
     at an interior extremum of `K` is the natural next attack, as is the
     hierarchy of bilinear constraints from integrating test functions
     against `K″` (the first one: `∬(x-y)ℓτ(x-y) dq(x)dp(y) = 0`).

  Lean deliverables: `hasDerivAt` of the displacement kernel (two-case, the
  quadratic bound `|h(1-e^{-|h|/τ})| ≤ h²/τ` at the base point), the `D_p′`
  identity, the classical ODE, the Wronskian crosswalk, and the headline
  conditional reduction `laplaceZeroDrift_imp_eq_of_companionAligned`.
  Axiom-free; arbitrary probability measures on `ℝ`; no moment hypotheses.
  Still open: proving (or refuting) `K ≡ 0` from zero drift alone.
- **Stage 3b (RESOLVED on the finite class, 2026-07-10): the atomic
  converse.**  THEOREM (new; numerically verified to 1e-17 at every step;
  formalization in `LaplaceAtomicConverse.lean`):

  > For every `τ > 0` and all finitely-supported probability measures
  > `p = Σ aᵢ δ_{zᵢ}`, `q = Σ bᵢ δ_{zᵢ}` on `ℝ` (common refined support
  > `z₁ < … < z_N`, arbitrary `N` and atoms), pointwise zero raw Laplace
  > mean-shift drift forces `p = q`.

  This is the first genuine arbitrary-PAIR converse content for the paper's
  practical kernel (Dirac rigidity had one degenerate side), and it resolves
  the authors' open question on the dense class of finite mixtures — exactly
  the representation class of the paper's own Appendix-C heuristic, with NO
  frame conditions, probe choices, or bandwidth restrictions.

  *Proof (complete).*  Zero drift is the bilinear identity
  `Φ(x) := Σᵢⱼ aᵢbⱼ(zᵢ-zⱼ)kτ(x,zᵢ)kτ(x,zⱼ) = 0` for all `x` (the certified
  cross-displacement form).  On each open interval between consecutive atoms
  (`z_k < x < z_{k+1}`, including the unbounded ends), each kernel factor is
  a one-sided exponential, so with `u := e^{2x/τ}` and
  `αᵢ := aᵢe^{zᵢ/τ}`, `βᵢ := bᵢe^{zᵢ/τ}`:

  `Φ(x)·u = 𝔞ₖ + 𝔟ₖ·u + 𝔠ₖ·u²`,   `𝔞ₖ = Σ_{i,j≤k} αᵢβⱼ(zᵢ-zⱼ)`,

  with constants `𝔞ₖ,𝔟ₖ,𝔠ₖ` depending only on the interval.  A quadratic
  polynomial vanishing on a nondegenerate interval of `u`-values is zero, so
  **`𝔞ₖ = 0` for every `k`** — a hierarchy of truncated antisymmetric moment
  constraints ("moment parallelism": the truncated tilted moment vectors
  `(Σ_{i≤k}αᵢzᵢ, Σ_{i≤k}αᵢ)` of `p` and `q` are parallel at every
  truncation).  These constraints alone force `b = a`:

  1. *Bottom atoms match.*  If `a` had no mass at the smallest atoms where
     `b` does (or vice versa), the first nonvacuous `𝔞ₖ` would be a strictly
     signed sum: `𝔞ₖ = αₖ Σ_{j<k} βⱼ(zₖ-zⱼ) > 0`.  So `a₁ > 0 ↔ b₁ > 0`,
     and both hold (the refined support is the union).  Set `λ := β₁/α₁`.
  2. *Telescoping induction.*  Assume `βₗ = λαₗ` for `l ≤ m`.  Substituting
     into `𝔞_{m+1} = 0` and using the antisymmetry of the double sum in a
     symmetric argument (`Σ_{j,l} αⱼαₗ(zⱼ-zₗ) = 0`),

     `0 = 𝔞_{m+1} = (β_{m+1} - λα_{m+1}) · Σ_{j≤m+1} αⱼ(zⱼ - z_{m+1})`,

     and the last factor is strictly negative (`α₁ > 0`, `zⱼ < z_{m+1}`), so
     `β_{m+1} = λα_{m+1}`.
  3. *Normalization.*  `bᵢ = λaᵢ` for all `i` and `Σb = Σa = 1` give
     `λ = 1`, hence `b = a` and `p = q`.  ∎

  Numerical verification (seed 7, `N = 5`, random atoms/weights): interval
  decomposition residual `1.4e-17`; induction identity residual `2.1e-17`;
  end-to-end reconstruction of `b` from the constraint family recovers `a`
  to `5.6e-17`.

- **Stage 3c (research, toward general measures):** en route to the atomic
  theorem, two identities were derived for ARBITRARY measures under zero
  drift by Stieltjes-differentiating the interval decomposition of
  `Φ ≡ 0` (the `dp`/`dq` coefficients cancel identically):
  `e^{-2x/τ}𝔞(x) = e^{2x/τ}𝔠(x)` for all `x` (with `𝔞(x)` the left-truncated
  bilinear pairing `∬_{y,y′≤x}(y-y′)e^{(y+y′)/τ}dp(y)dq(y′)`), and the
  measure identity `m·(Z_p dq - Z_q dp) = -(2/τ)𝔟 dx` (`m` the common mean
  shift) — so on `{m ≠ 0}` the cross difference `Z_p dq - Z_q dp` is
  absolutely continuous.  The general-measure analogue of the atomic proof
  would follow from `𝔞 ≡ 0` (then `d𝔞 = 0` gives the measure identity
  `Q(x)dp = P(x)dq` with `P,Q` the one-sided compensated moments, and a
  Grönwall/uniqueness argument should close); whether zero drift forces
  `𝔞 ≡ 0` beyond the atomic class is the sharpened open question, alongside
  the alignment identity `K ≡ 0` of Stage 3.

- **Stage 4 (research): general `d`** via subordination (mixture of Gaussian
  bandwidths) — can the Gaussian converse be applied bandwidth-wise under the
  mixture? The obstruction: zero MIXED drift does not obviously give zero
  per-bandwidth drift; look for a positivity/interlacing argument.

## Non-goals and honesty

- No claim that the full converse is resolved: Stage 1 is a structural
  reduction (new, load-bearing, machine-checked), not the identification.
- The asymptotic `V → 0` question stays out of scope (see
  `RawFieldConverse.md` Part I §5).
- Any failed proof route gets logged in `LoggedFailures.md` as usual.
