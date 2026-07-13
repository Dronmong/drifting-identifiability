# Conditional axiom implementation plan for the 1-d Laplace a.c. converse

Date: 2026-07-11 (revised after a codebase audit; status updated 2026-07-12)

> STATUS 2026-07-12: the finite-simple-zero a.c. derivation route is now
> machine-checked and axiom-free.  `LaplaceACFinal.lean` proves the
> continuous-density finite-simple-zero theorem, and
> `LaplaceACGaussianCertificate.lean` proves the standard-Gaussian
> non-vacuity witness.  This axiom-route document is therefore **not** the active
> implementation plan for the proved finite-simple-zero theorem.  It is retained
> only as a FALLBACK design for a stronger future theorem: unrestricted
> a.c. + exponential-moment targets with no finite-zero/sign-pattern hypothesis.
> Do not implement any scaffold axiom below unless the project explicitly
> decides to replace that broader analytic bridge by a conditional external
> theorem.

Purpose: give future agents a precise, auditable, and TOOLING-ACCURATE fallback
route for turning the **broader** paper-level absolutely-continuous /
exponential-moment proof (LaplaceGeneralConverseRoadmap.md, Milestone 5) into an
opt-in conditional Lean module without cheating.  It should not be used for the
already-proved continuous-density finite-simple-zero theorem.

This revision corrects an earlier draft that (a) proposed trivially-inhabited
predicates, which silently reduce the "conditional" theorem to the forbidden
unconditional one; (b) misdescribed the trust tooling, whose actual constraints
make the integration harder than "add a file and classify it"; and (c)
under-rated the multiple-zero case, which is a genuine open gap, not a review
formality. Each correction is called out below.

Convention: prose and all axiom/theorem IDENTIFIERS are ASCII (the trust scripts
and PowerShell logs match on ASCII names). Lean TYPE signatures use the
project's standard Unicode notation (`ℝ`, `τ`, `∀`, `≪`), because these blocks
are meant to be copy-pasteable into real project files and the audit scanner
keys only on the declaration keyword and name, never on the types.

## Executive summary

The project has already machine-checked the downstream algebraic gate:

```text
(forall x, laplaceKernelNormalizerWronskian tau p q x = 0)  ->  p = q
```

as `laplaceKernelNormalizer_wronskian_eq_zero_imp_eq`
(DriftingIdentifiability/LaplaceGeneralConverseWronskian.lean:219), where

```text
W(x) = laplaceKernelNormalizerWronskian tau p q x
     = (2/tau) * (Pplus(x) * Qminus(x) - Pminus(x) * Qplus(x))
```

(mass-determinant form: `laplaceKernelNormalizerWronskian_eq_massDet`, same file
line 94). The one missing analytic step for a.c. measures is:

```text
ACExpMoment1D tau p
ACExpMoment1D tau q
MeanShiftSingleCrossing1D tau p            (Phase 3A only)
ZeroDrift (meanShiftDrift (laplaceKernel tau)) p q
--------------------------------------------------------
forall x, laplaceKernelNormalizerWronskian tau p q x = 0
```

The deliverable is to put THIS step (and only this step) behind an explicit,
contentful, hash-pinned conditional axiom, then compose it in one line with the
certified gate.

DO NOT axiomatize `ZeroDrift ... p q -> p = q`. That axiomatizes the open
problem. CRITICALLY: with trivially-inhabited predicates, the Phase-3A axiom
BECOMES this forbidden axiom (see Soundness rule 1). The two soundness rules
below are not style advice; violating either makes the module unsound or
pointless.

## Non-negotiable soundness rules

### Rule 1 -- predicates MUST carry mathematical content

An earlier draft proposed:

```lean
structure ACExpMoment1D (tau : ℝ) (mu : Measure ℝ) : Prop where
  external_regular : Prop        -- WRONG
```

A field of type `Prop` holds a proposition, not a proof of one, so
`ACExpMoment1D tau mu` is inhabited for EVERY `tau, mu` by `⟨True⟩`, and
`∀ tau mu, ACExpMoment1D tau mu` is provable. The same holds for any
single-`Prop`-field `MeanShiftSingleCrossing1D`. Consequently the Phase-3A axiom
could be instantiated at ANY `p, q`, and the derived theorem collapses to

```text
ZeroDrift (meanShiftDrift (laplaceKernel tau)) p q  ->  p = q     (arbitrary p, q)
```

which is the forbidden unconditional statement -- and, since the arbitrary
converse is UNKNOWN for singular-continuous measures, this risks importing an
actually-false axiom (global inconsistency: `False` becomes provable).
Documentation cannot fix this; it is a logical fact about the predicate.

Therefore the predicates must have real content from the first commit. Concrete
required shape:

```lean
/-- External analytic regularity class for the 1-d a.c. Laplace converse:
absolute continuity plus two-sided exponential first moments. NOT a placeholder;
`ac` is load-bearing -- it is what makes the axiom unprovable at Diracs and hence
genuinely conditional. -/
structure ACExpMoment1D (τ : ℝ) (μ : Measure ℝ) : Prop where
  ac       : μ ≪ (volume : Measure ℝ)
  mom_pos  : Integrable (fun y => Real.exp (y / τ)) μ
  mom_neg  : Integrable (fun y => Real.exp (-y / τ)) μ
  mom1_pos : Integrable (fun y => |y| * Real.exp (y / τ)) μ
  mom1_neg : Integrable (fun y => |y| * Real.exp (-y / τ)) μ
```

With `ac : μ ≪ volume` present, `ACExpMoment1D τ (Measure.dirac a)` is
UNPROVABLE. That is the entire point: it is what separates the conditional axiom
from the forbidden one.

For the single-crossing restriction, encode the actual geometry (the 1-d mean
shift is strictly antitone, hence crosses zero at most once and transversally):

```lean
/-- Single mean-shift crossing: `m_p(x) = meanShift (laplaceKernel τ) p x`
(a real number in 1-d) is strictly antitone. Combined with the tail asymptotics
supplied by `ACExpMoment1D` (m_p → +∞ at −∞, → −∞ at +∞) this gives exactly one
zero, at which m_p changes sign. -/
structure MeanShiftSingleCrossing1D (τ : ℝ) (p : Measure ℝ) : Prop where
  strictAnti : StrictAnti (fun x => meanShift (laplaceKernel τ) p x)
```

(The exact minimal form of this predicate should be finalized against the proof;
`StrictAnti` is the intended content. Do NOT reduce it to a bare `Prop` field.)

### Rule 2 -- the axiom's HOME and its trust wiring

The audit (scripts/TrustAudit.ps1) enforces, today:

- Any `axiom` / `constant` / `opaque` declared OUTSIDE `Paperaxioms.lean` is a
  hard audit FAILURE (TrustAudit.ps1:101-108).
- `Paperaxioms.lean` is gated by BOTH a hard-coded allowlist
  (`$paperEstablishedAxioms` + `$conditionalExternalAxioms`, lines 11-37) AND a
  SHA256 manifest `.trusted/Paperaxioms.sha256` (lines 39-51).
- Only `TrustedBoundary.lean` may `import DriftingIdentifiability.Paperaxioms`
  directly (lines 110-113).
- The default root module may not import `Conditional` /
  `CharacteristicIdentifiability` / `GaussianNondegeneracy` (lines 66-69).

CORRECTION to the earlier draft (and to the first-pass audit): the axiom CANNOT
be placed in `Paperaxioms.lean`, because its conclusion mentions
`laplaceKernelNormalizerWronskian`, which is defined far DOWNSTREAM of
`Paperaxioms.lean` (in LaplaceGeneralConverseWronskian.lean). Pulling that
vocabulary up into `Paperaxioms.lean` is not feasible. So a separate axiom file
is FORCED by the dependency structure -- but the current audit forbids exactly
that. The earlier draft's "add a file, then update TrustAudit to classify it"
understates the work: the audit's notion of "designated axiom file" is
hard-wired to a single path and must be GENERALIZED.

Required trust-tooling change (a deliberate trust-boundary extension, needs the
same human sign-off as any Paperaxioms edit):

1. Add a new opt-in axiom file, e.g.
   `DriftingIdentifiability/LaplaceACExternal.lean`.
2. Generalize `scripts/TrustAudit.ps1` from a single hard-coded axiom file to a
   REGISTRY of designated axiom files, each with its own allowlist and its own
   hash manifest:

   ```text
   $axiomFiles = @(
     @{ path='Paperaxioms.lean';     allow=$allowedPaperAxioms;        hash='.trusted/Paperaxioms.sha256' },
     @{ path='LaplaceACExternal.lean'; allow=$conditionalAnalyticAxioms; hash='.trusted/LaplaceACExternal.sha256' }
   )
   ```

   - The "forbidden axiom outside Paperaxioms" check becomes "forbidden axiom
     outside ANY registered axiom file."
   - Each registered file gets its own allowlist Compare-Object and its own
     Get-FileHash pin.
   - Extend the root-import guard so the root tree cannot reach
     `LaplaceACExternal` (add it, and the conditional Laplace module, to the
     forbidden-root-import regex).
3. Create `.trusted/LaplaceACExternal.sha256` and regenerate it on every
   approved change:

   ```powershell
   (Get-FileHash -Algorithm SHA256 `
     DriftingIdentifiability/LaplaceACExternal.lean).Hash `
     | Out-File -Encoding utf8 .trusted/LaplaceACExternal.sha256
   ```

Rationale: the reason axioms are confined to one file is auditability -- one
hash-pinned, allow-listed surface. A registry preserves that guarantee for a
second file instead of quietly weakening the "axioms only in Paperaxioms" rule.
This is strictly better than the alternative of special-casing the audit to wave
one file through.

## Current trust status

Machine-checked, main-track facts (all axiom-free):

- `laplaceKernelNormalizer_wronskian_eq_zero_imp_eq`: `W ≡ 0 -> p = q`
  (LaplaceGeneralConverseWronskian.lean:219; the composition target).
- `laplaceKernelNormalizer_wronskian_zero_imp_eq`: same gate, unfolded form
  (line 147).
- `laplaceKernelNormalizerWronskian_eq_massDet`: closed mass-determinant formula
  (line 94).
- `laplaceKernelNormalizerWronskian_continuousWithinAt_Ici`: right-continuity of
  `W` (line 371).
- `laplaceKernelNormalizerWronskian_eq_zero_of_right_tail` / `_of_left_tail`:
  `W = 0` in the tails from support facts (lines 343, 355) -- these can shrink
  the axiom surface (see Phase 3A refinement).
- `laplaceZeroDrift_identifies_of_truncatedPairing_eq_zero`: `frakturA ≡ 0 ->
  p = q`.
- finite atomic and right-dense-gap support classes.

Paper-level, NOT Lean-certified:

- the a.c. + exponential-moment proof (distributional Green identities, corrected
  tail asymptotics, Abel/Wronskian evolution, asymptotic ODE uniqueness).
  Numerically corroborated by `numerics/milestone5_wronskian.ps1`.

Existing conditional convention to MATCH (do not reinvent):

- Conditional Gaussian/RKHS axioms already live in `Paperaxioms.lean` and are
  classified `$conditionalExternalAxioms` in TrustAudit.ps1 (lines 29-35).
- The theorems consuming them live in `CharacteristicIdentifiability` /
  `GaussianNondegeneracy`, aggregated by `Conditional.lean`, kept out of root.
  The new module follows the SAME aggregation and root-exclusion pattern; it
  differs only in needing a second registered axiom file (Rule 2).

Operational notes:

- `lake build DriftingIdentifiability` is fast and green.
- `scripts/TrustAudit.ps1` (hash + allowlist) is the FAST gate that governs
  adding an axiom. `scripts/AxiomAudit.ps1` (`#print axioms` per declaration) is
  a SEPARATE, slower gate that may time out as the promoted list grows; if it
  does, audit new declarations individually rather than weakening it. Do not
  conflate the two.

## The scaffold axiom is not a "standard external result"

Project policy (see auto-memory `axiomatize-well-known-theorems`): axiomatize
STANDARD external results; PROVE the project's own open results. This bridge
axiom is the project's OWN unformalized theorem, not a textbook citation, so it
does not sit comfortably beside `equation_6_loss_value`. Treat it as a clearly
marked, TEMPORARY research scaffold:

- give it its own audit category (`$conditionalAnalyticAxioms`), never fold it
  into `$paperEstablishedAxioms`;
- its docstring must say "conditional analytic scaffold; paper-proved, not yet
  Lean-formalizable; to be discharged via Phase 4";
- the principled destination is Phase 4 (Levinson asymptotic integration, Abel's
  identity, dominated-convergence tail limits -- those ARE standard external
  results), with the bridge DERIVED in Lean. Phase 3A buys a useful theorem now;
  Phase 4 is what eventually removes the scaffold.

## Literature orientation

The Phase-4 external facts are standard analytic territory:

- 1-d Laplace / double-exponential kernel = Green kernel of a 2nd-order
  constant-coefficient operator.
- Abel's identity: Wronskian evolution for a 2nd-order linear ODE.
- Levinson / Hartman-Wintner asymptotic integration: tail-mode uniqueness when
  the variable-coefficient ODE approaches `Z'' = Z / tau^2`.

References:

- N. Levinson, "The asymptotic nature of solutions of linear systems of
  differential equations", Duke Math. J. 15 (1948), 111-126.
- Hartman-Wintner, "Asymptotic integrations of linear differential equations",
  Amer. J. Math. 77 (1955), 45-87.
- Eastham, *The Asymptotic Solution of Linear Differential Systems*, LMS
  Monographs, 1989.
- Abel's identity: https://en.wikipedia.org/wiki/Abel%27s_identity

## Axiom route ratings (revised)

Usefulness = Lean/time saved toward a conditional identifiability theorem.

| Route | Truth status | Usefulness | Recommendation |
|---|---|---|---|
| Green-function identities (4.1) | standard, safe | low alone | Principled but insufficient alone; needs distribution machinery. |
| Tail asymptotics (4.2) | standard (DCT), safe | medium | Provable in Lean with effort; part of the destination. |
| Abstract ODE uniqueness (4.3) | standard (Levinson) | medium-high | Best long-term boundary; connecting it to the normalizers needs 4.1/4.2. |
| A.c. single-crossing bridge (3A) | paper-solid | high | SHIP FIRST. Contentful predicates mandatory. Scaffold axiom. |
| A.c. general bridge (3B) | NOT known-true | high | HOLD -- do not axiomatize (see multiple-zero gap). |
| Direct `ZeroDrift -> p=q` | forbidden / unknown | -- | Never. Trivial predicates silently reduce 3A/3B to THIS. |

## Recommended phased implementation

### Phase 0 -- documentation and trust boundary

Preserve the existing separation: `LaplaceArbitraryConverse.md`,
`ResearchStatus.md`, and `LaplaceGeneralConverseRoadmap.md` keep the a.c. case
labelled paper-level. Add a one-line pointer from `ResearchStatus.md` to this
plan and to `numerics/milestone5_wronskian.ps1`.

### Phase 1 -- the opt-in module and its wiring

Single opt-in file, holding the predicates, the scaffold axiom, AND its sole
consumer (co-locating axiom and consumer maximizes auditability):

```text
DriftingIdentifiability/LaplaceACExternal.lean
```

Wiring:

- Register it as a designated axiom file (Rule 2): generalize TrustAudit,
  add `$conditionalAnalyticAxioms`, add `.trusted/LaplaceACExternal.sha256`.
- Import it ONLY through `DriftingIdentifiability/Conditional.lean`.
- Reach `laplaceKernelNormalizerWronskian` and the gate by importing
  `DriftingIdentifiability.LaplaceGeneralConverseWronskian` (which already brings
  the Laplace + `ZeroDrift` / `meanShiftDrift` vocabulary transitively). Do NOT
  write `import DriftingIdentifiability.Paperaxioms` -- that line fails the audit
  for any non-boundary file.
- Do NOT import it from `DriftingIdentifiability.lean`. Extend the root-import
  guard to forbid it explicitly.

### Phase 2 -- contentful predicates

Add `ACExpMoment1D` and `MeanShiftSingleCrossing1D` exactly as in Rule 1
(contentful, no bare-`Prop` fields). Do not overbuild further measure theory
before the conditional theorem exists, but the `ac`/moment/monotonicity content
is the MINIMUM, not optional.

Optionally consider taking `ZeroDriftAE` (a.e. zero drift,
TrustedBoundary.lean:36) instead of pointwise `ZeroDrift`: it is the physically
natural hypothesis (energy = 0), and `zeroDrift_of_ae_of_continuous_fullSupport`
(TrustedBoundary.lean:62) already upgrades a.e. -> pointwise for full-support
laws (e.g. Gaussians). This yields a stronger, better-motivated axiom at no cost
for the full-support a.c. case. Decide per whether bounded-support a.c. laws
(uniform) are in scope.

### Phase 3A -- SHIP THIS: single-crossing conditional theorem

Why this is the safe first target -- and why it is genuinely paper-solid, not
just "less delicate":

- With one mean-shift zero `z*`, BOTH sides of `z*` are SEMI-INFINITE intervals.
  The tail/Abel argument forces the analytic Wronskian to vanish on each open
  side independently (no reliance on propagation through a singular point).
- a.c. => atomless => `W` is continuous at `z*`. Right-continuity is certified
  (`laplaceKernelNormalizerWronskian_continuousWithinAt_Ici`); the matching
  atomless left-continuity follows from the mass-determinant form plus
  Milestone-1 continuity of the one-sided masses (a small companion lemma worth
  adding). Hence `W(z*) = 0` and `W ≡ 0`.

This is exactly the argument the multiple-zero case CANNOT run (next section).

Scaffold axiom (in `LaplaceACExternal.lean`, registered per Rule 2):

```lean
/-- CONDITIONAL ANALYTIC SCAFFOLD (not a paper citation, not Lean-certified).
For a.c. laws with two-sided exponential moments and a single mean-shift
crossing, pointwise zero mean-shift drift forces the raw-normalizer Wronskian to
vanish identically. Paper-proved: LaplaceGeneralConverseRoadmap.md, Milestone 5.
Not yet formalizable (Mathlib lacks Levinson-type asymptotic ODE integration and
distributional elliptic identities). Numerically corroborated:
numerics/milestone5_wronskian.ps1. To be discharged via Phase 4. -/
axiom laplaceAC_singleCrossing_zeroDrift_wronskian_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : ACExpMoment1D τ p) (hq : ACExpMoment1D τ q)
    (hcross : MeanShiftSingleCrossing1D τ p)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) :
    ∀ x : ℝ, laplaceKernelNormalizerWronskian τ p q x = 0
```

Conditional theorem (one-line composition with the certified gate; verified to
typecheck against the real signature at LaplaceGeneralConverseWronskian.lean:219):

```lean
theorem laplaceAC_singleCrossing_identifies_conditional
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : ACExpMoment1D τ p) (hq : ACExpMoment1D τ q)
    (hcross : MeanShiftSingleCrossing1D τ p)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) :
    p = q :=
  laplaceKernelNormalizer_wronskian_eq_zero_imp_eq τ hτ p q
    (laplaceAC_singleCrossing_zeroDrift_wronskian_zero τ hτ p q hp hq hcross hzero)
```

Optional surface-shrinking refinement: state the axiom's conclusion only on the
support interior, and reach `∀ x` in Lean via the certified tail lemmas
`laplaceKernelNormalizerWronskian_eq_zero_of_right_tail` / `_of_left_tail` (for
bounded-support a.c. laws) and the gap-constancy lemmas. This moves more of the
argument inside certified Lean and narrows what the axiom asserts. Weigh against
the extra statement complexity; not required for a first cut.

### Phase 3B -- HOLD: broader a.c. bridge (multiple zeros)

CORRECTION to the earlier draft, which rated this "safety medium, needs review."
The multiple-zero case is a GENUINE open gap, and axiomatizing it would risk
importing a false axiom. Do not ship it as an axiom until the gap is closed on
paper.

The gap, precisely (this also patches an over-confident line in the roadmap):
across an interior mean-shift zero `z_k` the ODE is singular (`m(z_k) = 0`), and
Abel's factor `exp(-∫ 2 mu'/m)` has a `~ C/(x - z_k)` singularity there. Whether
it tends to 0 or to infinity is governed by `sign(mu'(z_k) / m'(z_k))` -- exactly
the Frobenius exponent `beta = M_p/(tau Z_p)` computed in
`numerics/milestone5_wronskian.ps1`. When it tends to 0, a nonzero interior
second-mode admixture survives WITHOUT violating continuity at `z_k`, and the
only remaining defense is the mass constraint `∫ d(q - p) = 0`. But positive
integrals of individual interior bumps do NOT preclude CANCELLATION between
bumps of opposite sign at different interior zeros. The roadmap's "killed by the
mass constraint" glosses this.

Do NOT accept a Phase-3B axiom on the strength of:

```text
interior second-mode bumps have positive integral, so total mass zero kills them
```

Before any 3B axiom, a written proof must establish ONE of:

- all interior second-mode coefficients share a sign (rules out cancellation); or
- positivity of `Z_q` forces each interior coefficient to vanish individually; or
- the Frobenius exponent has the "confining" sign at every interior zero
  (the `-> infinity` branch), so continuity alone forces each interior constant
  to zero; or
- the number of zeros is bounded by an explicit added hypothesis, reducing to a
  finite gluing.

Until then, `MeanShiftSingleCrossing1D` (Phase 3A) is the honest boundary.

### Phase 4 -- the principled destination (lower-level standard axioms)

These are the genuinely-standard external results; discharging the scaffold means
proving `laplaceAC_singleCrossing_zeroDrift_wronskian_zero` FROM these in Lean.
Reprioritized above the earlier draft's "optional": these are what make the
development honest w.r.t. the project's axiom policy.

- 4.1 Green identities: `(1 - τ² ∂²) Z_μ = 2 τ μ`,
  `(1 - τ² ∂²) D_μ = 2 τ² Z_μ'` (weak/distributional form). Very safe; needs weak
  derivatives in Lean.
- 4.2 Tail asymptotics: `exp(x/τ) Z_μ(x) -> ∫ exp(y/τ) dμ`; tilted mean -> tilted
  first moment; `m_μ(x) = mu_tilt(x) - x ~ const - x`. Likely provable by
  dominated convergence under the `ACExpMoment1D` moment fields.
- 4.3 Abstract ODE uniqueness (Levinson): a theorem about REAL FUNCTIONS and ODE
  coefficients only -- no measures, no `ZeroDrift`, no `p = q`. That is what makes
  it a genuine imported analytic theorem rather than a disguised project result.
  Connecting it to the normalizers will require 4.1/4.2.

## Legitimacy / anti-triviality obligations

The project already ships the exact meta-checks; USE them (the earlier draft did
not). All in `TrustedBoundary.lean`:

- `ConditionAllowsDistinctPair` (line 83): "rules out conditions that merely hide
  `p = q` in their assumptions."
- `IsLegitimateCondition` (line 99): `(∃ p q, condition p q) ∧
  ConditionAllowsDistinctPair condition`.
- pattern to copy: `bothProbability_allowsDistinctPair` (line 90).

Required before accepting Phase 3A:

1. Prove `∃ μ, ACExpMoment1D τ μ` by exhibiting a Gaussian (rules out the VACUOUS
   failure mode: an empty predicate makes the axiom useless).
2. Prove `ConditionAllowsDistinctPair` for the effective condition by exhibiting
   two DISTINCT Gaussians both satisfying the predicate (rules out the
   HIDES-`p=q` failure mode).
3. Note explicitly (reviewer check, not a theorem): `ACExpMoment1D` must LITERALLY
   contain `μ ≪ volume`, which rules out the TOO-WEAK failure mode of Rule 1. No
   mechanical check catches an over-weak predicate; a human must read the
   definition and confirm the content is present.

Failure modes 1 and 2 are caught by `IsLegitimateCondition`; failure mode 3 is
caught only by inspecting the definition. All three must be checked.

## Trust-audit integration (concrete, corrected)

1. Put predicates + axiom + conditional theorem in
   `DriftingIdentifiability/LaplaceACExternal.lean`.
2. Generalize `scripts/TrustAudit.ps1` to a registry of designated axiom files
   (Rule 2); add `$conditionalAnalyticAxioms =
   @('laplaceAC_singleCrossing_zeroDrift_wronskian_zero')`; extend the
   root-import guard to forbid the new module from the root tree.
3. Create and populate `.trusted/LaplaceACExternal.sha256` (regen command in
   Rule 2). This hash bump IS the human review checkpoint.
4. Import the module only through `DriftingIdentifiability/Conditional.lean`.
5. Do NOT edit `DriftingIdentifiability.lean`.
6. Do NOT add the conditional theorem to the paper-native promoted list in
   `scripts/AxiomAudit.ps1`; if a conditional section is wanted there, keep it
   separate from paper-native decls.
7. Build and audit:

   ```powershell
   lake build DriftingIdentifiability.LaplaceACExternal
   lake build DriftingIdentifiability.Conditional
   lake build DriftingIdentifiability
   powershell -ExecutionPolicy Bypass -File scripts/TrustAudit.ps1
   ```

8. Per-declaration axiom check (bypasses AxiomAudit timeout):

   ```lean
   import DriftingIdentifiability.LaplaceACExternal
   #print axioms DriftingIdentifiability.laplaceAC_singleCrossing_identifies_conditional
   ```

   Expected: Lean foundations + the single named scaffold axiom. NOT the Gaussian
   / RKHS conditional axioms, and NOT any final-conclusion identifiability axiom.

9. REGRESSION check (the earlier draft omitted this): confirm the MAIN,
   non-conditional headline theorems still print only
   `propext / Classical.choice / Quot.sound`. Structurally guaranteed by the
   root-exclusion, but cheap insurance against accidental contamination.

## Acceptance criteria

Phase 3A:

- New conditional module builds; `lake build DriftingIdentifiability` still green.
- Predicates are CONTENTFUL (Rule 1): `ACExpMoment1D` contains `μ ≪ volume` and
  the four moment integrals; `MeanShiftSingleCrossing1D` contains the
  `StrictAnti` content. No bare-`Prop` fields.
- Legitimacy obligations 1-3 discharged (Gaussian existence + distinct-pair +
  content inspection).
- The axiom is in the registered `LaplaceACExternal.lean`, allow-listed as
  `$conditionalAnalyticAxioms`, and hash-pinned.
- The axiom's conclusion is `W ≡ 0`, never `p = q`.
- `p = q` is a one-line composition with the certified Wronskian gate.
- The theorem name contains `conditional`; the axiom docstring says "scaffold,
  paper-proved, not Lean-certified" and cross-references the numerics.
- `#print axioms` on the conditional theorem shows exactly the one scaffold axiom;
  regression check on main theorems passes.

Phase 3B: all of the above, PLUS a written proof resolving the multiple-zero
cancellation (one of the four bullet options), reviewed before the axiom lands.
Absent that proof, 3B does not ship.

## Red flags for reviewers

Reject or revise any implementation that:

- introduces `axiom ... : ZeroDrift ... -> p = q`;
- uses a bare-`Prop`-field (trivially inhabited) `ACExpMoment1D` or
  `MeanShiftSingleCrossing1D` -- this SILENTLY equals the forbidden axiom
  (Rule 1), regardless of documentation;
- omits `μ ≪ volume` from `ACExpMoment1D`;
- declares the axiom outside a REGISTERED, hash-pinned axiom file, or "classifies"
  it in TrustAudit without generalizing the file registry and regenerating the
  hash;
- writes `import DriftingIdentifiability.Paperaxioms` in a non-boundary file;
- imports the conditional module into the default root, or fails to extend the
  root-import guard;
- names the scaffold axiom as if it were paper-native or Lean-certified, or folds
  it into `$paperEstablishedAxioms`;
- skips the `IsLegitimateCondition` obligations;
- ships Phase 3B without the written multiple-zero proof;
- claims the full arbitrary-measure converse from the a.c. axiom.

## Recommended immediate next step

SUPERSEDED by the derivation route (`LaplaceACDerivation.md`): rather than
implement the 3A scaffold axiom, prove `W ≡ 0` from zero drift directly. Fall
back to the Phase-3A axiom below ONLY if a derivation step (in particular
tilted-mean monotonicity, step 4) proves infeasible; in that case implement it
with contentful predicates, the registered + hash-pinned scaffold axiom
(`W ≡ 0` conclusion), the one-line conditional theorem, and the
`IsLegitimateCondition` obligations. Hold Phase 3B regardless until the
multiple-zero cancellation is resolved on paper.
