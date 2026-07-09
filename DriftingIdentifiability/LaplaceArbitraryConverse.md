# Laplace-kernel converse for arbitrary targets: attack plan and research record

**Status:** Stage 1 in progress (2026-07-09). This file records the plan and
the mathematical findings BEFORE implementation, per project practice, so the
track is resumable from this document alone.

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

- **Stage 1 (THIS session, Lean, axiom-free, promoted):** new module
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
  Register in the default root + promoted axiom audit; full Check.ps1;
  `#print axioms` foundations-only.
- **Stage 2 (next): Laplace smoothing injectivity + Dirac rigidity.**
  Formalize `Z_p = Z_q → p = q` (via `charFun`/Fourier positivity of the
  Poisson profile, or in 1-d via the elliptic inversion) and the
  constant-conditional-mean ⟹ Dirac mini-theorem.
- **Stage 3 (research): the 1-d ODE/Wronskian program** recorded above, aiming
  at the full 1-d arbitrary-target converse (or a counterexample — the
  fundamental-system structure near zeros of `m` is where either a proof or a
  construction will come from).
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
