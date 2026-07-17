# L5 (n = 3 radial Laplace converse) — session handoff

> **2026-07-16: L5 is COMPLETE (see §0/§5e).  The next-phase plan — the
> general ℝⁿ converse (radial general n, slack removal, n = 2, and the
> foliation–cancellation route to arbitrary measures) — lives in
> `LaplaceRnRoadmap.md`.  Start there; this file remains the map of the
> closed n = 3 milestone and its five implementation files.**

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

**Target theorem — PROVED 2026-07-16 (`LaplaceRadialInvariance3.lean`; see §5e):**
```
theorem laplaceZeroDrift_identifies_of_radialMixture₃
    (τ : ℝ) (hτ : 0 < τ) (νp νq : Measure ℝ)
    [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)          -- supported on [0,∞)
    (hmp : Integrable id νp) (hmq : Integrable id νq)      -- first moments
    (hslackp : RadialSlack₃ τ νp)                          -- νp only; see §5(F6)
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
| **L5 R-layer (STARTED, definitions/bridges green)** | `DriftingIdentifiability/LaplaceRadialRay3.lean` |
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
  Generic substitution through `shellDistSubst` is done
  (`integral_comp_shellDistSubst_mul_deriv`, positive-orientation variant), and
  `shellT` has its factored z-form
  `(4r²s)⁻¹∫z(r²+s²-z²)e^{-z/τ}dz`.  The non-collision and removable collision
  `ρ²/d` z-forms are done, and the full positive-radius T₃ theorem is certified:
  `shellRhoSqOverDist_eq_two_tau_div_r_mul_shellT`.
  **Current S-layer location:** prove the ray-mass identity.
- **Update 2026-07-15:** `LaplaceRadialRay3.lean` now exists and builds.  It
  defines `radialRayZ₃`, `radialRayC₃`, `radialRayD₃`, `radialRayT₃`, and
  `radialRayRhoSqOverDist₃`, plus bridge lemmas to the shell-layer normalizer,
  companion-normalizer, and drift-coordinate formulas.  It intentionally does
  not yet assert C¹/system identities.
- **Update 2026-07-15 (second wave — R-LAYER COMPLETE, SYSTEM STARTED):**
  `LaplaceRadialRay3.lean` now contains, all green and committed:
  - **The full C¹ layer**: pointwise probe-derivatives away from collisions
    (`hasDerivAt_norm_rayProbe_sub`, `hasDerivAt_laplace{,Companion}Kernel_rayProbe'`,
    `hasDerivAt_laplaceKernel_mul_coord'`), the collision-null lemma
    `radialMixture₃_ae_probe_ne`, and the three dominated-differentiation
    theorems **`hasDerivAt_radialRayZ₃`** (`Z̃' = (1/τ)Z̃d`),
    **`hasDerivAt_radialRayC₃`** (`C̃' = (1/τ)D̃`),
    **`hasDerivAt_radialRayD₃`** (`D̃' = (1/τ)Q̃ − Z̃`), with y-level payloads
    `radialRayZd₃ = ∫K(X/d)`, `radialRayQ₃ = ∫K(X²/d)`.
  - **The closure identity** `radialRayC₃_closure`:
    `C̃ = Q̃ + 3τZ̃ + (2τ/r)D̃`, via `integral_chartBase_zonal_measurable`
    (measurable-G φ-collapse), the bridges `radialRayT₃_eq_integral` /
    `radialRayRhoSqOverDist₃_eq_integral`, the mixture T₃
    `radialRayRhoSqOverDist₃_eq_T` (`P̃ = (2τ/r)T̃`), and
    `radialRayT₃_eq_D_add_Z` (`T̃ = D̃ + rZ̃`).
  - **The sign layer**: `radialRayZ₃_pos`, `abs_radialRayZd₃_le`,
    `radialRayM₃`/`radialRayMDeriv₃`, `hasDerivAt_radialRayM₃`,
    `radialRayMDeriv₃_cov`, **`radialRayMDeriv₃_ge_of_le`** (the PROVED
    `m̃ ≤ r ⟹ m̃' ≥ −3` case — via pointwise AM–GM `2K|X| ≤ K(X²/d)+Kd`,
    NO Cauchy–Schwarz), **`RadialSlack₃`** (the named far-tilt hypothesis),
    `radialRayMDeriv₃_ge` (combined).
  `LaplaceRadialSystem3.lean` (NEW) has the zero-drift foundation:
  `zeroDrift_ray_D_mul_Z` (`D̃_pZ̃_q = D̃_qZ̃_p`), `zeroDrift_ray_M_eq` /
  `zeroDrift_ray_MDeriv_eq` (common `m̃`, `m̃'`), `zeroDrift_D_deriv_bridge`
  (covariance bridge by `HasDerivAt.unique`), system objects
  `radialRayW₃/V₃/K₃/Khat₃`, and `zeroDrift_C_eq` (the `g_ν`-form
  `C̃_ν = τm̃'Z̃_ν + m̃Z̃d_ν + 4τZ̃_ν + (2τ/r)m̃Z̃_ν`).

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
   The polynomial FTC core, reverse-substitution algebra, set-to-interval bridge,
   generic substitution lemma, `shellT` z-form, collision regularization, and
   full positive-radius T₃ theorem are DONE.  Remaining S-layer item: the
   ray-mass identity `∫₀^∞ r²Z̄(r,s)dr = 2τ³`.

### 5b. `LaplaceRadialRay3.lean` (R-layer) — **DONE (2026-07-15, all green)**
C¹ layer, closure identity, sign layer all committed; see the update bullet in
§4.  Nothing remains here.

### 5c. `LaplaceRadialSystem3.lean` — **STARTED (foundation green)**
DONE: zero-drift ray reduction, common `m̃`/`m̃'`, covariance bridge,
`radialRayW₃/V₃/K₃/Khat₃` defs, `zeroDrift_C_eq` (the `g_ν`-form).
NEXT (in order — the first two are pure algebra over `zeroDrift_C_eq`):
1. **`radialRayKhat₃_eq`: `K̂ = τ·m̃·v`** under zero drift (`hr : 0 < r`).
   Proof sketch: unfold defs; `rw [zeroDrift_C_eq …(νp), zeroDrift_C_eq …(νq),
   D=M·Z-facts]; field_simp [hr.ne', hτ.ne']; ring`.  (`hM` for νp is
   `fun _ => rfl`-trivial; for νq it is `zeroDrift_ray_M_eq`.)
2. **`hasDerivAt_radialRayKhat₃`: `K̂' = −τ(m̃'+4)·v`.**  Assemble the
   HasDerivAt for `fun r => r²(C̃_pZ̃_q − C̃_qZ̃_p)` from the R-layer
   derivative theorems by product rule (`(hasDerivAt_id r).pow 2`,
   `.mul`, `.sub`), then rewrite the derivative VALUE into `−τ(m̃'+4)v` via
   `zeroDrift_C_eq` (both measures) + `D = M·Z` + `field_simp; ring` (the
   full paper computation is in the §4.10 status-log bullet "system file
   started" — it is literally linear in the `g_ν`-facts).
3. `|K̂| ≤ C·r²` near 0 (`|C̃| ≤ τ`, `|Z̃| ≤ 1`, so `|K̂| ≤ 2τr²`) and
   `K̂ → 0` at ∞ under first moments (`Integrable id ν`; split `s ≶ r/2`,
   `r·tail → 0` by dominated convergence — see §4.10(F8)).
4. The **trichotomy** (F7) reusing `LaplaceACPropagation.lean` — mirror
   `LaplaceUnconditionalConverse.lean:882–1010` (the `m̃(x₀){=,<,>}0` case
   split, `sInf`/`sSup` of the zero set, edge lemmas
   `abel_{left,right}_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_{lower,upper}`
   and `abel_{right,left}_outer_zero_of_muDeriv_nonneg_of_m_{neg,pos}`),
   with `μ̂ := (m̃'+4)/2 ≥ ½` from `radialRayMDeriv₃_ge` and the `r = 0`
   universal edge (`m̃ ≤ L·r` near 0 via local `m̃'`-bounds + MVT).
   W-slot ← `radialRayKhat₃`; m-slot ← `radialRayM₃ τ νp`; muDeriv-slot ←
   `fun t => (radialRayMDeriv₃ τ νp t + 4)/2` (matches the wrappers'
   hard-coded `2μ/m` coefficient exactly).
5. Endgame steps 1–3: `K̂≡0 ⟹ v≡0` (on `{m̃≠0}` divide; on `int{m̃=0}` use
   `K̂' = 0 = −4τv`; glue with `continuous_eq_zero_of_dense_zeroSet`,
   `LaplaceACPropagation.lean:1343`) `⟹ (Z̃_p/Z̃_q)' = 0 ⟹ Z̃_p = c·Z̃_q`,
   then `c = 1` via the ray-mass identity (5a.2, still to prove).

### 5d. STATUS UPDATE 2026-07-16 — TRICHOTOMY AND RAY ENDGAME ARE DONE
`LaplaceRadialConverse3.lean` (all green, committed through the trichotomy;
the ray-endgame commit may be pending if the session ended during a tool
outage — check `git status`):
- continuity layer, Abel shape, three edge cores, origin-edge providers,
  **`radialRayKhat₃_eq_zero`** (K̂ ≡ 0 on (0,∞)),
  **`radialRayV₃_eq_zero`** (v ≡ 0),
  **`radialRayZ₃_proportional`** (`Z̃_p = c·Z̃_q` on (0,∞)).

### 5e. STATUS UPDATE 2026-07-16 (later) — **L5 IS COMPLETE**

`LaplaceRadialInvariance3.lean` (builds green) closes everything that was
listed as remaining in 5d:

1. **Invariance is DONE** — the θ-flow argument went through exactly as
   designed, with one improvement: the collision shells (`s = ‖x‖`) did NOT
   need the countable-atoms/dense-radii argument.  The tilted per-shell
   identity extends across the collision by **s-continuity at fixed x**
   (`chartBase_tilted_eq_shellZ'`: both sides are continuous in `s`, equal
   for `s ≠ R`, conclude with `tendsto_nhds_unique_of_eventuallyEq` on
   `𝓝[≠] R`).  The master theorem is
   **`kernelNormalizer_radialMixture₃_radial`**: `Z_ν(x) = Z̃_ν(‖x‖)` for
   EVERY probe `x` (`x = 0` via `rayProbe_zero_eq`; `x ≠ 0` via spherical
   angles `θ₀ = arccos(x₀/‖x‖)`, `ψ = Complex.arg⟨x₁,x₂⟩`).
2. **`c = 1` is DONE, and the Tonelli/ray-mass identity was NEVER NEEDED**
   (that plan item is deleted).  Discovery: L4's `LaplaceSmoothingInjective`
   is stated for **finite** measures, so after extending proportionality to
   `r = 0` (**`radialRayZ₃_proportional'`**, right-limit continuity), apply
   `laplaceSmoothingInjective_euclideanSpace (ι := Fin 3)` to the pair
   `(radialMixture₃ νp, (ENNReal.ofReal c) • radialMixture₃ νq)` — the
   normalizer hypothesis holds exactly by `integral_smul_measure`.  That
   yields `p = (ofReal c)•q`; evaluating at `univ` forces `ofReal c = 1`
   (`ENNReal.ofReal_eq_one`), and `one_smul` finishes.
3. **THE HEADLINE IS PROVED**: `laplaceZeroDrift_identifies_of_radialMixture₃`
   (hypotheses exactly as in §0, with only νp's slack, as predicted).

Root wiring (all five L5 files added to `DriftingIdentifiability.lean`) and
`AxiomAudit.ps1` entries (12 promoted L5 declarations, headline included)
are in place.  Remaining bookkeeping only: keep `Check.ps1` green and commit.

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
