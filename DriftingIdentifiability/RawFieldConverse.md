# Raw-field general converse: landscape record + axiom-route plan

**Status:** planning document, written 2026-07-08. Part I records where the
identifiability question stands across the paper, the authors' reviewer
rebuttal, and this project. Part II is the resumable work plan for the open
piece — the general converse for the *raw* mean-shift drift field of arbitrary
targets — using the **axiom route** (deliberate: this is a discovery/
contribution project, not a formalize-everything project; well-known external
facts are axiomatized with citations, per the standing project policy).

Main track (not the Sinkhorn extension). Follows the candidate discipline in
`AGENTS.md` / `CandidateConditions.lean`: legitimacy check → counterexample
thought → written proof → Lean theorem.

---

## Part I. The landscape: paper vs rebuttal vs us

The theoretical claim at issue is the **converse / identifiability** of the
drifting field: does vanishing drift force the two laws to coincide? Written
for the paper's mean-shift field (equations 8/10):

```
meanShift k p x    = Z_p(x)⁻¹ · ∫ k(x,y)(y−x) dp(y)          -- eq 8
meanShiftDrift k p q x = meanShift k p x − meanShift k q x   -- eq 10  ("V")
Z_p(x) := kernelNormalizer k p x = ∫ k(x,y) dp(y)
```

The target predicate is `IdentifiesAtZero condition (meanShiftDrift k)`, i.e.
`∀ p q, condition p q → (∀ x, V p q x = 0) → p = q` (`TrustedBoundary.lean`).

### 1. The paper (arXiv:2602.04770v2)

Gives a **heuristic** identifiability argument in Appendix C.1: represent `p, q`
by finite coefficients (eq 29), expand the drift bilinearly (eqs 30–31), and
claim a nondegeneracy that forces equal coefficients hence `p = q`. It is a
sketch — the nondegeneracy is asserted, not proved, and there is no rigorous
statement of scope. This is precisely the "theory gap (W3)" a reviewer flagged.

### 2. The authors' reviewer rebuttal (the screenshot)

Two **informal analytic** arguments, "planned for the revised paper":

- **Rebuttal argument 1 (Gaussian kernel, general targets).** With a Gaussian
  kernel the mean-shift field equals `σ²[∇ₓ log p̂_σ(x) − ∇ₓ log q̂_σ(x)]`
  (score difference of the Gaussian-smoothed densities). If `V ≡ 0` then
  `p̂_σ = q̂_σ`, and Gaussian convolution is injective, so `p = q`.
- **Rebuttal argument 2 (Laplacian kernel, Gaussian targets).** For
  `k(x,y)=exp(−‖x−y‖/τ)` with `P,Q` Gaussian, the field probed along `x = ru`
  satisfies `V(ru) → (μ_p−μ_q) + (Σ_p−Σ_q)u/τ` as `r→∞`, determining both mean
  and covariance for the Gaussian family.

They explicitly concede: *"The general converse for arbitrary fields remains
open."*

### 3. What we have (this project)

Rigorous, machine-checked, and quantitative — the only non-heuristic artifact
in the picture. Three relevant pieces:

- **Trusted route (default, no conditional axioms).** The finite-basis
  interaction-frame / Vandermonde converse: for a finite probability-density
  basis whose induced interaction vectors satisfy a positive frame bound,
  `V = 0` at the selected probes forces `p = q` (`finitePopulationMeanShift_
  identifies`; frame discharged axiom-free for explicit Gaussian point-mass
  families in `EmpiricalFrameBound.lean` and a `C∞` bump basis). Crucially it is
  **exact and quantitative**: `‖a−b‖₁ ≤ (2B/c)‖V_probes‖`, plus a finite-sample
  bridge. Neither the paper nor the rebuttal has the quantitative form. Machine
  check: 160 promoted declarations, *no* conditional Gaussian/RKHS dependencies.

- **Conditional characteristic route (opt-in, `DriftingIdentifiability.
  Conditional`).** `characteristicKernel_identifiesAtZero` /
  `gaussianMmd_identifiesAtZero` (`CharacteristicIdentifiability.lean`) are
  **rebuttal argument 1's mechanism, made rigorous modulo one cited axiom**.
  BUT they are stated for the **MMD field** `mmdDrift kg p q x = ∫kg(x,·)dp −
  ∫kg(x,·)dq` (an *unnormalized* embedding difference), not the raw normalized
  `meanShiftDrift`. They rest on `characteristic_gradientEmbedding_injective`
  and `gaussian_gradient_isCharacteristic` (Sriperumbudur et al. 2010,
  axiomatized) and are explicitly labeled "reductions, not accepted project
  solutions."

- **Conditional asymptotic route.** `gaussianMmd_asymptoticallyIdentifies`:
  MMD-discrepancy → 0 implies weak convergence (Lévy–Prokhorov), via the
  metrization axiom `gaussian_mmd_metrizes_weakConvergence`. Again the *MMD
  discrepancy*, not the raw field norm.

### 4. Head-to-head

| claim | paper | rebuttal | us |
|---|---|---|---|
| exact `V≡0 ⟹ p=q`, finite/structured basis | heuristic | — | **proved, axiom-free, quantitative** |
| exact `V≡0 ⟹ p=q`, Gaussian, arbitrary target | — | arg 1 (informal) | **only for the MMD field, and conditional** — the raw-field version is the gap this plan closes |
| asymptotic `V→0 ⟹ qₙ→p` | — | arg 1/2 (informal) | conditional, MMD discrepancy only |
| Gaussian/Laplacian moment recovery (arg 2) | — | arg 2 (informal) | **not present** |
| estimator-level finite-sample theory | — | — | **present (Algorithm 2 SNIS etc.)** |

### 5. The honest open problem (agreed by everyone)

The general converse for the **raw** drift field of **arbitrary** targets:
`V ≡ 0 ⟹ p = q` for `meanShiftDrift k`, general `p, q`. Two facts frame it:

- **A kernel condition is unavoidable.** For a band-limited kernel (Fourier
  transform vanishing on a set) one can build `p ≠ q` with identical smoothed
  fields, so `V ≡ 0` while `p ≠ q`. "Arbitrary target" is reachable; "arbitrary
  kernel" is provably false. The realistic statement is *characteristic kernel,
  arbitrary target*.
- **The raw field is normalized, and that is the technical crux** (see Part II).
  The asymptotic (`V→0`) version is strictly harder and may be false as literally
  stated for the raw field: kernel smoothing kills high-frequency differences, so
  `‖V‖` can be tiny while `qₙ ↛ p` (`LoggedFailures.md`). The asymptotic version
  needs a metrizing discrepancy (MMD) or tightness hypotheses and is **out of
  scope** for this plan, which targets the exact `V≡0` case only.

---

## Part II. Plan for the open piece (axiom route)

**Goal.** A promoted (opt-in/conditional) theorem: for the Gaussian kernel and
**arbitrary** probability measures, `meanShiftDrift (gaussianKernel σ) ≡ 0 ⟹
p = q`. This closes the gap between "what we proved (MMD field)" and "the raw
mean-shift field the algorithm's population idealization actually uses," and
makes rebuttal argument 1 rigorous **modulo one clearly-scoped, cited external
axiom** — which is exactly the posture we want (stronger than the rebuttal,
which leans on the same fact informally).

### The technical crux (why the existing axioms do NOT already give this)

The existing conditional route closes the **MMD field** in one line because
`mmdDrift = ∫kg dp − ∫kg dq` **is** the embedding difference, so
`mmdDrift = 0 ⟺ embeddings match`, and `characteristic_gradientEmbedding_
injective` fires directly.

The raw field is different. Using `equation_11_bilinear_mean_shift`,
`meanShiftDrift k p q x = (Z_p(x) Z_q(x))⁻¹ · ∫∫ k(x,y₁)k(x,y₂)(y₁−y₂) d(p×q)`,
and since `Z_p, Z_q > 0` (from `MeanShiftRegularAt`),

```
V ≡ 0  ⟺  A_p(x)·Z_q(x) = Z_p(x)·A_q(x)  for all x,   A_p(x) := ∫ k(x,y) y dp
       ⟺  A_p(x)/Z_p(x) = A_q(x)/Z_q(x)              (normalized conditional means match)
```

This is `meanShift k p = meanShift k q` (immediate from the definition too:
`V = meanShift p − meanShift q`). It is **not** `A_p = A_q` — the MMD/embedding
condition. The two differ by the per-law normalizers `Z_p, Z_q`. So MMD
embedding injectivity does not apply as-is; the raw converse needs injectivity of
the **normalized conditional-mean map** `p ↦ meanShift k p`. For the Gaussian
this is exactly the score/convolution-injectivity fact of rebuttal argument 1
(the normalizer is the smoothed density, and `meanShift ∝ ∇log(p∗φ_σ)`), but as
a statement it is one honest axiom, not a corollary of what we already have.

### The axiom (the one new external fact)

Add to `Paperaxioms.lean`, in the conditional-external class (with the existing
characteristic axioms; cite Sriperumbudur et al. 2010 for Gaussian
characteristicness + the standard score-determines-density step):

```lean
/-- Well-known consequence of Gaussian characteristicness (Sriperumbudur et al.
2010) plus score-determines-density: the normalized Gaussian mean-shift map is
injective on probability measures.  For the Gaussian kernel, `meanShift` is the
score of the Gaussian-smoothed law (`∝ ∇log(p∗φ_σ)`); matching scores forces
equal smoothed densities, and Gaussian convolution is injective, so `p = q`.
This constrains the mean-shift map only; it does not mention a drifting field or
its zeros. -/
axiom gaussianMeanShift_injective
    {E : Type u} [MeasurableSpace E] [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [CompleteSpace E] [FiniteDimensional ℝ E]
    [BorelSpace E] [SecondCountableTopology E]
    (σ : ℝ) (hσ : ValidBandwidth σ)
    (p q : Distribution E) [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (h : ∀ x, meanShift (gaussianKernel σ) p x = meanShift (gaussianKernel σ) q x) :
    p = q
```

Notes for whoever implements:
- Keep the hypothesis about the **`meanShift` map**, not the drift — so the axiom
  is manifestly *not* the identifiability conclusion in disguise (same discipline
  as `characteristic_gradientEmbedding_injective`; the docstring must say so).
- Prefer this single bundled axiom over trying to reuse `mmdDrift`: the
  normalization mismatch above means there is no clean `meanShiftDrift ⟹ mmdDrift`
  reduction, so reusing the MMD axioms would require inventing the missing bridge
  anyway. One honest axiom is cleaner than a fake reduction.
- If a more general-kernel statement is wanted later, mirror the existing
  `IsCharacteristic` abstraction: introduce `MeanShiftCharacteristic k : Prop`
  and an injectivity axiom parameterised by it, with the Gaussian as one
  witnessing axiom. Secondary deliverable; do the Gaussian first.

### The theorem (one-line bridge, in the Conditional module)

Add to `CharacteristicIdentifiability.lean` (opt-in), reusing `BothProbability`:

```lean
/-- **Raw-field identifiability for the Gaussian mean-shift drift.**  Zero raw
mean-shift drift (equation 10) between two probability measures forces equality.
Closes the gap left by `gaussianMmd_identifiesAtZero`, which handled only the
unnormalized MMD field.  Rests on the single external fact
`gaussianMeanShift_injective`. -/
theorem gaussianMeanShiftDrift_identifiesAtZero (σ : ℝ) (hσ : ValidBandwidth σ) :
    IdentifiesAtZero (BothProbability (E := E)) (meanShiftDrift (gaussianKernel σ)) := by
  rintro p q ⟨hp, hq⟩ hzero
  haveI := hp; haveI := hq
  refine gaussianMeanShift_injective σ hσ p q ?_
  intro x
  -- meanShiftDrift = meanShift p − meanShift q, so zero drift ⟹ the maps agree
  exact sub_eq_zero.mp (hzero x)
```

`ZeroDrift V p q` unfolds to `∀ x, V p q x = 0`, and
`meanShiftDrift k p q x = meanShift k p x − meanShift k q x` by definition, so
`hzero x : meanShift … p x − meanShift … q x = 0` and `sub_eq_zero.mp` gives the
hypothesis of the axiom. (If the definitional unfolding does not fire directly,
`simp only [meanShiftDrift] at hzero` first.)

### Legitimacy obligation (mandatory, per `AGENTS.md`)

The condition must be shown not to secretly encode `p = q`. `BothProbability`
already has `bothProbability_allowsDistinctPair` (two distinct Dirac masses) —
reuse it; the target is `ConditionAllowsDistinctPair (BothProbability)`, already
proved. No new legitimacy proof needed, but the plan must *cite* it so the
reviewer sees the condition is nonvacuous. Optionally add an
`IsExactCounterexample` note recording the band-limited-kernel obstruction from
Part I §5 (shows why the kernel hypothesis cannot be dropped).

### Audit / build checklist

- [ ] `gaussianMeanShift_injective` added to `Paperaxioms.lean`, docstring states
      it is about the map, not the drift; cite Sriperumbudur et al. 2010.
- [ ] `TrustAudit.ps1` / the axiom classification updated: the conditional
      external axiom class grows by one (currently 5 → 6). Verify the Check.ps1
      banner count and any hash/allowlist the audit maintains.
- [ ] `gaussianMeanShiftDrift_identifiesAtZero` added to the **opt-in**
      `CharacteristicIdentifiability.lean` (NOT the trusted route), and registered
      in `scripts/AxiomAudit.ps1` promoted-declaration list.
- [ ] `#print axioms gaussianMeanShiftDrift_identifiesAtZero` shows exactly the
      three foundational axioms **plus** `gaussianMeanShift_injective` (and
      whatever `equation_11`/`meanShift` pull in) — and confirm it is reachable
      only through `DriftingIdentifiability.Conditional`, so the trusted-route
      audit still reports "no conditional Gaussian/RKHS dependencies".
- [ ] `scripts/Check.ps1` green.
- [ ] Update `ResearchStatus.md` (conditional-modules section) and this file's
      status line.

### Scope statement to carry into any write-up

This deliverable proves the **exact** `V ≡ 0 ⟹ p = q` converse for the **raw**
Gaussian mean-shift field and **arbitrary** targets, resting on one cited
external injectivity axiom (`gaussianMeanShift_injective`). It is the rigorous
form of the authors' rebuttal argument 1, upgraded from the MMD field to the
actual mean-shift field. It deliberately does **not**: (a) discharge the
injectivity from Fourier/Bochner (axiom route by choice); (b) address the
asymptotic `V → 0` version (needs a metrizing discrepancy or tightness — see
Part I §5); (c) cover non-characteristic kernels (provably false). Rebuttal
argument 2 (Laplacian/Gaussian moment recovery) remains a separate, not-yet-
attempted instance.

### Optional follow-ups (recorded, not in this deliverable)

- Abstract `MeanShiftCharacteristic k` marker + general-kernel injectivity axiom;
  Gaussian as a witness. Gives the general-kernel raw-field converse.
- Rebuttal argument 2 as a Lean instance: an asymptotic `x = ru, r→∞`
  moment-recovery theorem for the Gaussian family under the Laplacian kernel
  (parametric family, not arbitrary targets).
- The genuinely novel prize: a *quantitative* stability modulus `‖V‖ small ⟹
  D(p,q) small` for a characteristic kernel — beyond paper and rebuttal (both
  existence-only), in the spirit of our finite-basis `(2B/c)` bound. Obstruction:
  the modulus is frequency-dependent and degrades, which is exactly why the clean
  asymptotic statement resists formalization.
