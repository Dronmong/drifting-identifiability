# Conditional axiom implementation plan for the 1-d Laplace a.c. converse

Date: 2026-07-11

Purpose: give future agents a precise, auditable route for turning the
paper-level absolutely-continuous/exponential-moment proof into an opt-in
conditional Lean module without cheating. This file is intentionally separate
from `LaplaceGeneralConverseRoadmap.md` so agents can review the proposed axiom
boundary in isolation.

This plan is ASCII-only on purpose. Several project scripts and PowerShell logs
are easier to audit when theorem statements are not rendered with fragile math
Unicode.

## Executive summary

The project has already machine-checked the important downstream gates:

```text
W == 0              -> p = q
frakturA == 0       -> p = q
K == 0 / V constant -> p = q
```

Here `W` is the raw-normalizer Wronskian

```text
W(x) = laplaceKernelNormalizerWronskian tau p q x
     = (2/tau) * (Pplus(x) * Qminus(x) - Pminus(x) * Qplus(x)).
```

The missing analytic step for absolutely continuous measures is:

```text
ACExpMoment1D tau p
ACExpMoment1D tau q
ZeroDrift (meanShiftDrift (laplaceKernel tau)) p q
------------------------------------------------
forall x, laplaceKernelNormalizerWronskian tau p q x = 0
```

The safest useful implementation is to put this missing step behind an explicit
conditional external analytic axiom, then compose it with the already-certified
Lean theorem:

```text
laplaceKernelNormalizer_wronskian_eq_zero_imp_eq
```

Do not axiomatize:

```text
ZeroDrift (...) p q -> p = q
```

That would directly axiomatize the desired conclusion.

## Current trust status

Machine-checked, main-track facts:

- `laplaceKernelNormalizer_wronskian_eq_zero_imp_eq`:
  `W == 0 -> p = q`.
- `laplaceKernelNormalizerWronskian_eq_massDet`:
  closed mass-determinant formula for `W`.
- `laplaceKernelNormalizerWronskian_continuousWithinAt_Ici`:
  right-continuity of `W`.
- `laplaceZeroDrift_identifies_of_truncatedPairing_eq_zero`:
  `frakturA == 0 -> p = q`.
- finite atomic and right-dense-gap support classes.

Paper-level but not Lean-certified:

- the absolutely-continuous plus exponential-moment proof using distributional
  Green identities, corrected tail asymptotics, Abel/Wronskian evolution, and
  asymptotic ODE uniqueness.

Known operational issue:

- `lake build DriftingIdentifiability` is fast and green in the current
  project state.
- Individual new Wronskian declarations have audited cleanly.
- The monolithic `scripts/AxiomAudit.ps1` may time out after the promoted list
  grew. If agents modify audit wiring, they should consider splitting the audit
  into chunks rather than weakening the audit.

## Literature orientation

The proposed external facts are standard analytic territory:

- The 1-d Laplace kernel / double-exponential profile is a Green-kernel-type
  exponential for a second-order constant-coefficient operator.
- Abel's identity gives Wronskian evolution for pairs of solutions to a
  second-order linear ODE.
- Levinson / Hartman--Wintner asymptotic integration supplies the kind of
  tail-mode uniqueness needed when the variable-coefficient ODE approaches
  `Z'' = Z/tau^2`.

Useful references for reviewers:

- Norman Levinson, "The asymptotic nature of solutions of linear systems of
  differential equations", Duke Math. J. 15 (1948), 111--126.
  https://projecteuclid.org/journals/duke-mathematical-journal/volume-15/issue-1/The-asymptotic-nature-of-solutions-of-linear-systems-of-differential/10.1215/S0012-7094-48-01514-2.short
- Hartman--Wintner, "Asymptotic integrations of linear differential equations",
  Amer. J. Math. 77 (1955), 45--87.
- Eastham, *The Asymptotic Solution of Linear Differential Systems*,
  London Mathematical Society Monographs, 1989.
- Abel's identity / Wronskian formula for second-order linear ODEs:
  https://en.wikipedia.org/wiki/Abel%27s_identity

## Axiom route ratings

Usefulness means "how much Lean/time this saves toward a conditional
identifiability theorem."

| Route | Safety | Usefulness | Recommendation |
|---|---:|---:|---|
| Green-function identities | 9.5/10 | 2.5/10 | Safe but insufficient alone. Prefer proving eventually. |
| Tail asymptotics | 9/10 | 4/10 | Safe and important. Also plausible to prove in Lean with effort. |
| Abstract ODE asymptotic uniqueness / Abel bridge | 7/10 overall, 8.5/10 for single-crossing | 7--8/10 | Best serious axiom boundary. |
| A.c. Wronskian bridge `ZeroDrift -> W == 0` | 5.5/10 | 9.5/10 | Most useful compact bridge; keep conditional external. |
| Direct a.c. identifiability `ZeroDrift -> p=q` | 1/10 | 10/10 | Forbidden; axiomatizes the goal. |

## Recommended phased implementation

### Phase 0: documentation and trust boundary

Already done:

- `LaplaceArbitraryConverse.md` says the a.c. case is paper-level, not
  Lean-certified.
- `ResearchStatus.md` separates Lean-certified gates from the paper proof.
- `LaplaceGeneralConverseRoadmap.md` records the safe conditional-axiom plan.

Future agents should preserve this distinction.

### Phase 1: create an opt-in conditional module

Suggested file:

```text
DriftingIdentifiability/LaplaceACExternal.lean
```

Alternative:

```text
DriftingIdentifiability/Conditional/LaplaceAC.lean
```

The current project has a flat `DriftingIdentifiability/Conditional.lean`
aggregator, so the lower-friction route is:

```text
DriftingIdentifiability/LaplaceACExternal.lean
```

and import it only through:

```text
DriftingIdentifiability/Conditional.lean
```

Do not import it from:

```text
DriftingIdentifiability.lean
```

unless the project explicitly chooses to promote conditional external analytic
theorems into the default root.

### Phase 2: define transparent predicates

Add a lightweight predicate first. Do not overbuild measure theory before the
conditional theorem exists.

Suggested Lean skeleton:

```lean
import DriftingIdentifiability.LaplaceGeneralConverseWronskian

open MeasureTheory Set Filter Topology

namespace DriftingIdentifiability

open Paper

/-- External analytic regularity class for the 1-d a.c. Laplace converse.

This is intentionally a `Prop` wrapper. It should eventually be expanded into:
absolute continuity with respect to Lebesgue measure, two-sided exponential
first moment, and enough smoothness/tail regularity for the ODE asymptotic
argument. -/
structure ACExpMoment1D (tau : R) (mu : Measure R) : Prop where
  -- Placeholder fields are acceptable initially if the theorem is explicitly
  -- external/conditional. Prefer expanding these fields over time.
  external_regular : Prop

/-- Single-crossing/nondegenerate mean-shift geometry for the safer first
conditional theorem. This is not part of the final desired conclusion; it is
an explicit restriction that avoids the multiple-zero interior-mode issue. -/
structure MeanShiftSingleCrossing1D (tau : R) (p : Measure R) : Prop where
  external_single_crossing : Prop

namespace ExternalAnalytic

-- axioms go here

end ExternalAnalytic

end DriftingIdentifiability
```

Important: if using placeholder predicates, documentation must say they are
external analytic hypotheses, not proved regularity classes.

Preferred eventual expansion of `ACExpMoment1D`:

```text
mu << volume
Integral exp(y/tau) dmu < infinity
Integral exp(-y/tau) dmu < infinity
Integral |y| * exp(y/tau) dmu < infinity
Integral |y| * exp(-y/tau) dmu < infinity
```

The first-moment weighted versions are useful because the proof discusses the
tilted mean, not only the normalizer.

### Phase 3A: safest first conditional theorem, single crossing

This is the preferred first implementation.

Why: it avoids the most delicate part of Claude's proof, namely multiple
mean-shift zeros and possible cancellation among interior second-mode pieces.

Proposed external axiom:

```lean
axiom ExternalAnalytic.laplaceAC_singleCrossing_zeroDrift_wronskian_zero
    (tau : R) (htau : ValidBandwidth tau) (p q : Measure R)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : ACExpMoment1D tau p) (hq : ACExpMoment1D tau q)
    (hcross : MeanShiftSingleCrossing1D tau p)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel tau)) p q) :
    forall x : R, laplaceKernelNormalizerWronskian tau p q x = 0
```

Then prove:

```lean
theorem laplaceAC_singleCrossing_identifies_conditional
    (tau : R) (htau : ValidBandwidth tau) (p q : Measure R)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : ACExpMoment1D tau p) (hq : ACExpMoment1D tau q)
    (hcross : MeanShiftSingleCrossing1D tau p)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel tau)) p q) :
    p = q :=
  laplaceKernelNormalizer_wronskian_eq_zero_imp_eq tau htau p q
    (ExternalAnalytic.laplaceAC_singleCrossing_zeroDrift_wronskian_zero
      tau htau p q hp hq hcross hzero)
```

Safety rating: high enough for a conditional module.

Usefulness rating: 7/10.

Review checklist before accepting:

- Does the single-crossing hypothesis clearly mean "the common mean shift has
  exactly one zero and changes sign there"?
- Does the axiom stop at `W == 0`, not `p=q`?
- Is the theorem imported only through the conditional aggregator?
- Does `TrustAudit.ps1` classify the axiom as conditional external?

### Phase 3B: broader a.c. Wronskian bridge

Only add this after the multiple-zero argument is reviewed carefully.

Proposed axiom:

```lean
axiom ExternalAnalytic.laplaceAC_zeroDrift_wronskian_zero
    (tau : R) (htau : ValidBandwidth tau) (p q : Measure R)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : ACExpMoment1D tau p) (hq : ACExpMoment1D tau q)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel tau)) p q) :
    forall x : R, laplaceKernelNormalizerWronskian tau p q x = 0
```

Then prove:

```lean
theorem laplaceAC_identifies_conditional
    (tau : R) (htau : ValidBandwidth tau) (p q : Measure R)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : ACExpMoment1D tau p) (hq : ACExpMoment1D tau q)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel tau)) p q) :
    p = q :=
  laplaceKernelNormalizer_wronskian_eq_zero_imp_eq tau htau p q
    (ExternalAnalytic.laplaceAC_zeroDrift_wronskian_zero
      tau htau p q hp hq hzero)
```

Safety rating: medium.

Usefulness rating: 9.5/10.

Review checklist before accepting:

- The multiple-zero interior-mode claim must be written in a form that prevents
  cancellation between several interior intervals.
- In particular, reviewers should verify one of:
  - all interior second-mode coefficients have the same sign;
  - positivity of `Z_q` forces each interior coefficient individually to vanish;
  - a stronger orthogonality/mass argument kills each mode separately;
  - the number of zeros is actually controlled by an added hypothesis.

Do not accept the broad axiom based only on the sentence:

```text
interior second-mode bumps have positive integral, so total mass zero kills them
```

Positive integrals alone do not rule out cancellation if coefficients can have
opposite signs.

### Phase 4: optional lower-level axioms

These are mathematically cleaner but save less immediate time.

#### 4.1 Green-function identities

Possible axiom family:

```lean
axiom ExternalAnalytic.laplace_green_normalizer_identity
  -- `(1 - tau^2 * d^2) Z_mu = 2 * tau * mu`,
  -- in an appropriate distributional or weak form.

axiom ExternalAnalytic.laplace_green_displacement_identity
  -- `(1 - tau^2 * d^2) D_mu = 2 * tau^2 * Z_mu'`,
  -- in an appropriate distributional or weak form.
```

These are very safe, but Lean needs distribution/weak derivative machinery to
use them. They are not recommended as the first conditional deliverable.

#### 4.2 Tail asymptotics

Possible axiom family:

```lean
axiom ExternalAnalytic.laplace_normalizer_atTop_asymptotic
  -- `exp(x/tau) * Z_mu(x) -> Integral exp(y/tau) dmu`.

axiom ExternalAnalytic.laplace_tiltMean_atTop_tendsto
  -- Tilted mean tends to the exponentially tilted first moment.

axiom ExternalAnalytic.laplace_meanShift_atTop_linear
  -- `m_mu(x) = mu_tilt(x) - x ~ const - x`.
```

These are likely provable with dominated convergence under the right moment
hypotheses. Prefer proving them later if we want to reduce external axioms.

#### 4.3 Abstract ODE uniqueness

This is the mathematically best abstraction, but the statement is harder to
design.

Desired shape:

```lean
axiom ExternalAnalytic.secondOrder_laplaceTail_unique_decaying_solution
    -- If two functions solve the same second-order ODE with coefficients
    -- satisfying the Laplace-tail hypotheses, and both have the decaying
    -- `exp(-|x|/tau)` asymptotics at the relevant endpoints, then their
    -- Wronskian vanishes on the regular component.
```

This theorem should not mention measures, `ZeroDrift`, or `p=q`. It should be
a theorem about real functions and ODE coefficients. That makes it a genuine
imported analytic theorem, not a disguised project-specific result.

The challenge is connecting this abstract theorem to the existing Lean
normalizers; that connection may require additional lower-level axioms anyway.

## Trust-audit integration

If a new conditional axiom module is added:

1. Update `scripts/TrustAudit.ps1` so the exact new axioms are classified as
   conditional external axioms.
2. Do not add conditional theorem names to the paper-native promoted theorem
   list in `scripts/AxiomAudit.ps1`, unless the audit script has a separate
   conditional section that allows them.
3. Update `DriftingIdentifiability/Conditional.lean` to import the new module.
4. Do not update `DriftingIdentifiability.lean`.
5. Run:

   ```powershell
   lake build DriftingIdentifiability.LaplaceACExternal
   lake build DriftingIdentifiability.Conditional
   lake build DriftingIdentifiability
   powershell -ExecutionPolicy Bypass -File scripts/TrustAudit.ps1
   ```

6. If `scripts/AxiomAudit.ps1` times out, audit the new declarations
   individually with:

   ```lean
   import DriftingIdentifiability.LaplaceACExternal
   #print axioms DriftingIdentifiability.laplaceAC_singleCrossing_identifies_conditional
   ```

   The expected axiom set should include Lean foundations plus the explicitly
   named external analytic axiom. It should not include conditional
   Gaussian/RKHS axioms or any final-conclusion identifiability axiom.

## Acceptance criteria

For Phase 3A:

- New conditional module builds.
- Axioms are in `ExternalAnalytic`.
- The theorem name contains `conditional`.
- The axiom conclusion is `W == 0`, not `p=q`.
- The final `p=q` proof is a one-line composition with the existing certified
  Wronskian gate.
- Trust audit classifies the external axiom.
- Documentation states this is a conditional theorem for a.c. exponential
  moment plus single-crossing measures.

For Phase 3B:

- All Phase 3A criteria.
- A written review of the multiple-zero case is added before or with the axiom.
- The review explicitly addresses cancellation among several interior
  second-mode components.

## Red flags for reviewers

Reject or revise any implementation that:

- introduces `axiom ... : ZeroDrift ... -> p = q`;
- imports the conditional module into the default root without discussion;
- names the external theorem as if it were paper-native or Lean-certified;
- makes `ACExpMoment1D` an empty or trivially inhabited condition without clear
  documentation;
- hides the external axiom behind an innocent theorem name;
- fails to classify the axiom in the trust audit;
- claims the full arbitrary-measure converse from the a.c. axiom;
- ignores the multiple-zero cancellation issue.

## Recommended immediate next step

Implement Phase 3A first:

```text
conditional a.c. + exponential moment + single-crossing theorem
```

This gives a substantial, useful conditional theorem while avoiding the
least-vetted part of Claude's broader proof. After that, ask a heavy-duty model
or a human analyst to review the multiple-zero argument before promoting Phase
3B.
