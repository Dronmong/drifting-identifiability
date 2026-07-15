# L5 (n = 3 radial Laplace converse) — session handoff

*Written 2026-07-15.  Read this top-to-bottom; it is self-contained.  It tells a
new agent exactly what the goal is, what is already proved, what remains, and
where every relevant lemma/plan lives.*

---

## 0. The one-paragraph picture

We are formalising, in Lean 4 / Mathlib (v4.32.0-rc1), the paper-faithful
**higher-dimensional Laplace-kernel converse** for "Generative Modeling via
Drifting" (`papers/2602.04770v2.pdf`, kernel Eq. (12) is the **ℓ²/Euclidean**
Laplace kernel `k(x,y)=exp(−‖x−y‖₂/τ)`).  The 1-d converse and the all-dim
Gaussian converse are already **closed**.  Milestones **L0–L4** of the ℓ²/radial
program are **closed**.  The remaining milestone is **L5**: for
rotation-invariant ("radial") probability measures on `ℝ³`, zero Laplace
mean-shift drift ⟹ `p = q`.  We are building L5 for `n = 3` first (v1).  This is
the last milestone standing between us and the headline theorem.

**Target theorem (to be produced):**
```
theorem laplaceZeroDrift_identifies_of_radialMixture₃
    (τ : ℝ) (hτ : 0 < τ) (νp νq : Measure ℝ)
    [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)          -- supported on [0,∞)
    (hmomp : Integrable id νp) (hmomq : Integrable id νq)  -- first moments
    (hslackp : RadialSlack₃ τ νp) (hslackq : RadialSlack₃ τ νq)  -- see §5(F6)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
              (radialMixture₃ νp) (radialMixture₃ νq)) :
    radialMixture₃ νp = radialMixture₃ νq
```
(`RadialSlack₃` is a named, numerically-verified, conjectured-true hypothesis
covering one open sign case; see §5(F6).  If it is later proved, drop it.)

---

## 1. Trust discipline (MANDATORY — do not violate)

- **No `axiom`, `sorry`, `opaque`, `native_decide`** outside `Paperaxioms.lean`.
  Standard external theorems get axiomatized in `Paperaxioms.lean` **only** with
  an allowlist + hash entry (see the `axiomatize-well-known-theorems` memory);
  but L5 needs no new axioms — everything is elementary.
- Promoted theorems must have `#print axioms` showing only
  `propext / Classical.choice / Quot.sound`.
- `Check.ps1` must stay green; Lean builds must be `--wfail`-clean (the
  `unusedVariables` / `unusedSectionVars` linters are **hard errors** here).
- After finishing L5, run the audit (`AxiomAudit`) + `Check.ps1` before claiming
  done.

---

## 2. Where everything lives

| What | File |
|---|---|
| **This handoff** | `DriftingIdentifiability/LaplaceL5_HANDOFF.md` |
| **Full design record + all discoveries** | `DriftingIdentifiability/LaplaceHigherDim.md` — **§4.10 is the L5 v1 architecture (discoveries F0–F11); §4.9 is the earlier general-`n` derivation (R1–R8); the "Status log" bullets at the top of §4.10 track progress** |
| **L5 S-layer (IN PROGRESS, builds green)** | `DriftingIdentifiability/LaplaceRadialShell3.lean` |
| L0 (displacement potential `∇ψ=D`) | `DriftingIdentifiability/LaplaceRadialFoundations.lean` |
| L1 (far field) | `DriftingIdentifiability/LaplaceRadialFarField.lean` |
| L3 (atom alignment) | `DriftingIdentifiability/LaplaceAtomAlignment.lean`, `LaplaceConeExtraction.lean` |
| L4 (**n-d smoothing injectivity, CLOSED, unconditional**) | `DriftingIdentifiability/LaplaceRadialFourier.lean` → `laplaceSmoothingInjective_euclideanSpace`; conditional shell in `LaplaceEuclideanInjectivity.lean` |
| 1-d converse + the Abel/propagation engine we **reuse** | `LaplaceUnconditionalConverse.lean` (esp. `:882–:1010`), `LaplaceACPropagation.lean`, `LaplaceACAbel.lean`, `LaplaceACRegularity.lean`, `LaplaceGeneralConverseCompanionWronskian.lean` |
| Kernel / drift definitions | `Paperaxioms.lean` (`laplaceKernel :250`, `kernelNormalizer :196`, `meanShift :202`, `meanShiftDrift :209`, `ZeroDrift :83`, `ValidBandwidth :844`) |
| Build/audit | `Check.ps1`, `scripts/…`, `DriftingIdentifiability.lean` (root import list) |
| RSI numerics scratch | `…/scratchpad/RsiScan.cs` (C#; run via PowerShell `Add-Type`) |

**Build one file fast:** `lake build DriftingIdentifiability.LaplaceRadialShell3`
(≈30–90 s once Mathlib is cached; the import of `LaplaceRadialFoundations`
dominates).  Filter output with
`… 2>&1 | Select-String -Pattern "error|Build completed|✖|warning:"`.

`LaplaceRadialShell3.lean` is **not yet wired into the root**
`DriftingIdentifiability.lean` — do that only once the whole L5 chain is green,
to avoid slowing the full build during development.

---

## 3. The mathematical architecture (why it works)

The full derivation is `LaplaceHigherDim.md §4.10 (F0–F11)`.  Compressed:

Radial measures on `ℝ³` are `radialMixture₃ ν` (§4.10 F1): push `ν ⊗ chartBase`
through `(s,u,φ) ↦ s·Φ(u,φ)`, where `Φ(u,φ)=(u,√(1−u²)cos φ,√(1−u²)sin φ)`
charts the sphere and `chartBase` is the uniform prob. measure on
`[−1,1]×[−π,π]` (Archimedes: at `n=3` the `u`-marginal is uniform).

Probe on the ray `x = r·e₁`.  Because `‖r·e₁ − s·Φ(u,φ)‖² = r²+s²−2rsu` is
**φ-independent**, every ray integral collapses to `∫ s, (½∫_{−1}^1 …du) dν`.
Define ray objects (all functions of `r>0`):
```
Z̃(r) = Z_{radialMixture₃ ν}(r·e₁)              (normalizer)
C̃(r) = companion normalizer   (matérn-3/2 profile (d+τ)e^{−d/τ})
D̃(r) = e₁-component of the drift numerator ∫ e^{−d/τ}(y − r·e₁) dμ
m̃(r) = D̃/Z̃                                    (common tilted displacement under zero drift)
```
Key **closed identity** (§4.10 F4), provable per-shell by pure algebra:
`C̃ = τ·D̃' + 4τ·Z̃ + (2τ/r)·D̃`.  This replaces the entire Laplacian/IBP layer.

Set `w̃ = Z̃_p'Z̃_q − Z̃_q'Z̃_p`, `v = r²·w̃`, `K = C̃_pZ̃_q − C̃_qZ̃_p`,
`K̂ = r²·K`.  Under zero drift (which gives `D̃_pZ̃_q = D̃_qZ̃_p`, i.e. a common
`m̃`), algebra yields the **Abel system** (§4.10 F5):
```
K̂ = τ·m̃·v ,      K̂' = −τ·(m̃'+4)·v .
```
On `{m̃≠0}`: `K̂' = −[(m̃'+4)/m̃]·K̂` — the SAME shape the 1-d propagation
wrappers consume, with `μ̂ := (m̃'+4)/2` in place of 1-d's `(m'+2)/2`.  So
`LaplaceACPropagation.lean`'s lemmas apply **unchanged**.

Sign layer (§4.10 F6): `m̃'+1 = (1/τ)·Cov_w(X, X/d)` (`X = t−r`,
`t = ⟨y,e₁⟩`).  Proved: `m̃'≥−3` (i.e. `μ̂≥½`) on the region `m̃≤r`
(Cauchy–Schwarz + the T₃ corollary `E_w[d]=E_w[X²/d]+2τ·(r+m̃)/r`); the zero
case is free.  Open only on `m̃>r` (the far-tilt regime) → carried as the named
hypothesis `RadialSlack₃` (numerics say it is TRUE with margin; §4.10 status log
2026-07-15).

Endgame (§4.10 F7–F9): trichotomy over `(0,∞)` with `r=0` a universal edge
(mirror `LaplaceUnconditionalConverse.lean:882–1010`), `K̂→0` at ∞ under first
moments, gives `K̂≡0 ⟹ v≡0 ⟹ Z̃_p = c·Z̃_q`; the **ray-mass identity**
`∫₀^∞ r²Z̄(r,s)dr = 2τ³` (s-independent) forces `c=1`; O(3)-invariance lifts
`Z̃_p=Z̃_q` on the ray to `Z_p=Z_q` on all of `ℝ³`; then **L4**
(`laplaceSmoothingInjective_euclideanSpace (ι:=Fin 3)`) concludes `p=q`.

---

## 4. What is DONE (all committed, all build green, axiom-clean)

Design + numerics:
- `LaplaceHigherDim.md §4.10` — the entire F0–F11 architecture (commit `3f80a9a`).
- **RSI numerics** (commit `6d384ca`): 72k random + 14k adversarial samples;
  finite-difference check confirms `m̃' = −1 + Cov_w(X,X/d)/τ` to 1e-10; the open
  region `m̃>r` has `Cov ≥ +0.04τ` (strictly safe); global worst is a single
  shell just behind the probe (`Cov ≈ −0.046τ`, 40× margin vs the `−2τ`
  requirement).  ⇒ `RadialSlack₃` is conjectured-true; ship it as a hypothesis.

Lean, in `LaplaceRadialShell3.lean` (commits `…S-layer part 1/2`):
- `sphereChart`, `rayProbe` + component `rfl` lemmas + `continuous_sphereChart`.
- `sphereChart_normSq = 1`; **`dist_sq_rayProbe_smul_sphereChart`**
  (`‖r·e₁−s·Φ(u,φ)‖² = r²+s²−2rsu`, the φ-independence heart).
- `chartBase` (+ `IsProbabilityMeasure`), `chartMap`, **`radialMixture₃`**
  (+ `IsProbabilityMeasure`).
- **`integral_radialMixture₃`** — master collapse
  `∫ f d(radialMixture₃ ν) = ∫ s ∫ w f(chartMap(s,w)) dchartBase dν`.
- **`integral_chartBase_zonal`** — the φ-collapse
  `∫ w G dchartBase = ½∫_{−1}^1 g` for bounded φ-independent `G`.
- `shellDist`, `shellZ`; `laplaceKernel_rayProbe_{nonneg,le_one}`,
  `continuous_laplaceKernel_rayProbe`, `laplaceKernel_rayProbe_chart`.
- **`kernelNormalizer_radialMixture₃`**: `Z̃_ν(r) = ∫ shellZ τ r s dν` — the ray
  normalizer as a ν-mixture of per-shell zonal kernel averages.
- **Update 2026-07-15:** `shellC`, `kernelNormalizer_companion_radialMixture₃`,
  `shellD`, and `laplaceWeightedDisplacement_coord_radialMixture₃` are also done,
  so the basic ray objects `Z̃`, `C̃`, and `D̃` all have ν-mixture formulas.
  The T₃ vocabulary is in place (`shellAxial`, `shellRhoSq`, `shellT`,
  `shellRhoSqOverDist`, `shellD = shellT - r·shellZ`), and the reverse-distance
  polynomial core is now certified: `shellRhoPoly`, endpoint vanishing,
  `P'(z)=4z(r²+s²-z²)`, and
  `∫ P(z)e^{-z/τ} dz = 4τ∫ z(r²+s²-z²)e^{-z/τ} dz` over `[|r-s|,r+s]`.
  The reverse-distance substitution algebra is also in place:
  `u(z)=(r²+s²-z²)/(2rs)`, `u'=-z/(rs)`, endpoints `u(|r-s|)=1`,
  `u(r+s)=-1`, distance recovery `shellDist r s (u(z))=z`, and the pullbacks
  for `ρ²` and `s·u(z)`.  The set-integral bridge is also done:
  `shellT_eq_intervalIntegral` and `shellRhoSqOverDist_eq_intervalIntegral`.
  **Current S-layer location:** apply `intervalIntegral.integral_comp_mul_deriv'`
  with `shellDistSubst` to prove the original T₃ shell identity, then prove the
  ray-mass identity.

---

## 5. What REMAINS (the to-do list, in order)

Follow `LaplaceHigherDim.md §4.10 (F3–F11)` and the file plan §4.10(F11).  Suggested
file split (each downstream file imports the previous):

### 5a. Finish `LaplaceRadialShell3.lean` (S-layer)
1. **`shellC` and `shellD` are DONE** (`kernelNormalizer_companion_radialMixture₃`
   and `laplaceWeightedDisplacement_coord_radialMixture₃`).  The original API
   scout is kept here as historical context only:
   ```
   noncomputable def shellD (τ r s : ℝ) : ℝ :=
     (1/2) * ∫ u in Ioc (-1:ℝ) 1, Real.exp (-(1/τ) * shellDist r s u) * (s*u - r)
   -- target:
   -- (∫ y, laplaceWeightedDisplacement τ (rayProbe r) y ∂(radialMixture₃ ν)) 0
   --   = ∫ s, shellD τ r s ∂ν
   ```
   - **Integrability of the vector drift numerator is PUBLIC:**
     `laplaceWeightedDisplacement_integrable τ hτ (radialMixture₃ ν) (rayProbe r)`
     (`LaplaceCompanion.lean:222`, any `[IsFiniteMeasure p]`; needs
     `[BorelSpace] [CompleteSpace] [SecondCountableTopology]`, all hold for
     `EuclideanSpace ℝ (Fin 3)`).  `laplaceWeightedDisplacement τ x y =
     laplaceKernel τ x y • (y − x)` (`LaplaceCompanion.lean:201`).
   - **Component-0 extraction** (commute eval past the Bochner integral): the
     coordinate CLM is `EuclideanSpace.proj (0 : Fin 3) : StrongDual ℝ …`
     (`Analysis/InnerProductSpace/PiL2.lean:284`), simp lemma
     `EuclideanSpace.proj_apply` (`proj i y = y i`).  Use
     `(ContinuousLinearMap.integral_comp_comm (EuclideanSpace.proj 0) hFint).symm :
     (EuclideanSpace.proj 0) (∫ F) = ∫ y, (EuclideanSpace.proj 0) (F y)`, then
     `simpa [EuclideanSpace.proj_apply]` ⇒ `(∫ F) 0 = ∫ y, (F y) 0`.
   - **The integrand's 0-component:** `(laplaceKernel τ (rayProbe r) y • (y −
     rayProbe r)) 0 = laplaceKernel τ (rayProbe r) y * (y 0 − r)` via
     `PiLp.smul_apply`, `PiLp.sub_apply`, `rayProbe_apply_zero`.  On a chart point
     `(s • sphereChart u φ) 0 = s * u` (`PiLp.smul_apply` + `sphereChart_apply_zero`),
     so it collapses to `exp(−(1/τ)·shellDist r s u)·(s·u − r)` (φ-independent).
   - **Bound for the zonal `hC`** (the 0-component of the drift vector, needed to
     invoke `integral_chartBase_zonal`): `|(v) 0| ≤ ‖v‖` and
     `‖laplaceKernel • (y − rayProbe r)‖ = d·e^{−d/τ} ≤ τ·e⁻¹` via the **PUBLIC**
     `mul_exp_neg_div_le hτ (ht : 0 ≤ t) : t·exp(−t/τ) ≤ τ·exp(−1)`
     (`LaplaceRadialFoundations.lean:73`).  (`norm_laplaceWeightedDisplacement_le`
     and `mul_exp_neg_le` in `LaplaceCompanion.lean` are **PRIVATE** — reprove the
     `≤ τ·e⁻¹` bound from `mul_exp_neg_div_le` instead.)  So `hC` uses `C = τ·e⁻¹`
     (or any `C ≥` that; `τ` also works).
   This has already been implemented using the `integral_radialMixture₃` +
   `integral_congr_ae` + `integral_chartBase_zonal` skeleton of
   `kernelNormalizer_radialMixture₃`.
2. **T₃ per-shell identity** and the **ray-mass identity** `∫₀^∞ r²Z̄ dr = 2τ³`.
   The polynomial FTC core and reverse-substitution algebra of the T₃ route are
   DONE.  Remaining T₃ route (§4.10 refinement bullet, 2026-07-15): use the
   **reverse polynomial**
   `d`-substitution `u(z) = (r²+s²−z²)/(2rs)` (NO √-singularity, no endpoint
   case split) via `intervalIntegral.integral_comp_mul_deriv'` with primitives
   `Pₖ' = zᵏe^{−z/τ}` (`k ≤ 4`), then per-exponential-atom `ring`.

### 5b. `LaplaceRadialRay3.lean` (R-layer)
Ray objects as honest functions of `r`; **C¹ layer by dominated differentiation
of the ν-mixture integrals** (global constant dominators — see §4.10 refinement
(a): `|∂_r e^{−d/τ}| ≤ 1/τ`, etc.; differentiate under `∫ dν` with
`hasDerivAt_integral_of_dominated_loc_of_deriv_le`); `Z̃>0`; `C̃' = D̃/τ`; the
zero-drift ray reduction `D̃_pZ̃_q=D̃_qZ̃_p`; the closure identity (F4); the
`m̃'`-formula and the sign layer (F6): the `m̃≤r` Cauchy–Schwarz case, the zero
case, and the `RadialSlack₃` definition + `m̃>r` bridge.

### 5c. `LaplaceRadialSystem3.lean`
`w̃, v, K, K̂`; the F5 identities; `|K̂|≤C·r²` near 0; `K̂→0` at ∞ (first
moments); the **trichotomy** (F7) reusing `LaplaceACPropagation.lean` — mirror
`LaplaceUnconditionalConverse.lean:882–1010` (the `m̃(x₀){=,<,>}0` case split,
`sInf`/`sSup` of the zero set, edge lemmas
`abel_{left,right}_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_{lower,upper}`
and `abel_{right,left}_outer_zero_of_muDeriv_nonneg_of_m_{neg,pos}`).  Endgame
steps 1–3 (`K̂≡0 ⟹ v≡0 ⟹ Z̃_p=cZ̃_q`, then `c=1` via the ray-mass identity;
`continuous_eq_zero_of_dense_zeroSet` is at `LaplaceACPropagation.lean:1343`).

### 5d. `LaplaceRadialInvariance3.lean`
`Z_p(x) = Z̃_p(‖x‖)` from O(3)-invariance of `radialMixture₃` (polar formula +
one 2-d change of variables — see §4.10 F9.4; Householder reflection transports
`‖x‖e₁ ↦ x`).  **Fallback if the CoV fights Lean (§4.10 F9.5):** half-line Green
uniqueness on `(1−τ²∂²)(rZ̃)` or a 1-d charFun argument on the odd extension —
both ray-only, no invariance needed.

### 5e. `LaplaceRadialConverse3.lean`
Assemble the headline `laplaceZeroDrift_identifies_of_radialMixture₃` (feed L4 at
the last step), add `AxiomAudit` entries, wire into root
`DriftingIdentifiability.lean`, run `Check.ps1`.

---

## 6. Concrete Lean gotchas already hit (save yourself the round-trips)

- **`μ̃` (μ + U+0303 combining tilde) is NOT a valid Lean identifier** →
  "expected token".  Use `ν` (or `μt`).  (Design docs keep `μ̃`; Lean uses `ν`.)
- `ℝ≥0∞` needs `open scoped ENNReal` (not in scope here); use `ENNReal.ofReal`.
- `PiLp.continuous_toLp` is found by **`fun_prop`**; matrix literal `![…]`
  continuity is NOT — prove it via `continuous_pi` + `fin_cases` + `change` (defeq
  through `Matrix.cons_val_{zero,one,two}`, all `rfl`) then `fun_prop`.
- `Continuous.comp` often needs the middle map given explicitly: `.comp (f := …)`.
- `setIntegral_const` yields `μ.real s • c` (NOT `(μ s).toReal • c`) — compute
  with `Real.volume_real_Ioc_of_le`, not `Real.volume_Ioc`.
- `rw [← Real.exp_zero]` rewrites the **first** `1` it finds (e.g. the `1` in
  `1/τ`!) — instead rewrite `Real.exp_zero` **in a hypothesis** (`rwa […] at h`).
- `HasFiniteIntegral.of_bounded` (dot form), not `hasFiniteIntegral_of_bounded`.
- `Measure.isProbabilityMeasure_map` (dot form).
- `integral_map` / `integral_prod` need the integrand's `AEStronglyMeasurable` /
  `Integrable`; for bounded continuous `f` on the (finite) `radialMixture₃`,
  `⟨cont.aestronglyMeasurable, HasFiniteIntegral.of_bounded …⟩`.
- `omit`/`set_option … in` must precede a docstring; `--wfail` treats unused
  vars/hypotheses as errors (drop them or `_`-prefix).

---

## 7. Recommended working rhythm

Build incrementally, **commit at every green checkpoint** (the user explicitly
wants recorded progress in case a session ends), and **append a dated bullet to
the `LaplaceHigherDim.md §4.10 status log`** after each discovery/advance.  Batch
several new lemmas per build (the ~30–90 s build cost is import-dominated, so
more content per compile amortises better) — but only when you're reasonably
confident, since one error reruns the whole file.  Block synchronously on builds
(memory `dont-stop-for-background-tasks`): run `lake build …` foreground with a
long timeout and continue through.

Effort estimate (from §4.10): S-layer finish ≈ ½–1 session; R-layer ≈ 1–2;
System ≈ 1–2; Invariance ≈ 1 (or fallback); Converse assembly ≈ ½.  Risk is
concentrated in (i) the C¹ dominated-differentiation layer (5b) and (ii) the
O(3)-invariance CoV (5d, has a ray-only fallback).
