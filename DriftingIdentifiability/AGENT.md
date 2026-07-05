# Agent protocol: drifting identifiability

## Objective

Find explicit, mathematically meaningful conditions under which a zero (or
vanishing) drifting field identifies the target distribution:

```text
exact:       Vₚ,q = 0  ⟹  p = q
asymptotic:  driftSize(p,qₙ) → 0  ⟹  distance(p,qₙ) → 0
```

The reverse implication `p = q ⟹ Vₚ,q = 0` is already established by
anti-symmetry. The converse is false for arbitrary fields and remains the
research problem.

## Mandatory reading before mathematical work

Read these sources completely, in order:

1. `papers/2602.04770v2.pdf`, especially Sections 3.2–3.5 and Appendix C.
2. `DriftingIdentifiability/Paperaxioms.lean`.
3. `DriftingIdentifiability/TrustedBoundary.lean`.
4. `DriftingIdentifiability/LoggedFailures.md`.
5. `DriftingIdentifiability/WrittenProof.md`.

Do not rely on a remembered or second-hand version of the paper’s equations.

## Trusted boundary

`Paperaxioms.lean` is the sole reviewed axiom boundary. Only
`TrustedBoundary.lean` may import it directly.

Never do any of the following without explicit user approval:

- add or modify an `axiom`, `constant`, or `opaque` declaration;
- change `Paperaxioms.lean` or `.trusted/Paperaxioms.sha256`;
- use `sorry`, `admit`, `sorryAx`, or an unfinished theorem stub;
- assume the desired implication, its contrapositive, or an equivalent lemma;
- introduce a condition that contains `p = q`, `V = 0 ⟹ p = q`, uniqueness of
  the zero-drift equilibrium, injectivity of the target map, or another hidden
  restatement of the conclusion;
- derive the theorem from `False`, an inconsistent local instance, or an
  assumption known to be impossible.
- use metaprogramming or unsafe declarations to manufacture proof terms or
  bypass the kernel-visible dependency graph.

If a standard external theorem is missing from Mathlib, first try to prove it.
If that is impractical, document the exact statement and source and ask the
user before extending the trusted boundary. Browsing a source does not itself
authorize a new axiom.

Run `scripts/Check.ps1` after every meaningful Lean change. The trust audit
rejects proof escapes, new axioms/constants, direct imports that bypass the
boundary, and unapproved changes to the paper axioms.

## What counts as a legitimate condition

A proposed condition must be independently meaningful before zero drift is
assumed. In particular:

1. It must not syntactically or semantically encode the conclusion.
2. It must admit at least one pair `p ≠ q`; formalize this with
   `ConditionAllowsDistinctPair` or provide an explicit witness.
3. Its hypotheses must be checkable independently of the desired theorem.
4. Every regularity assumption must be stated: probability normalization,
   support, measurability, integrability, kernel positivity/characteristicness,
   topology of convergence, and finite-dimensionality where applicable.
5. Explain why the condition is plausible for the paper’s actual kernel and
   not merely for a redesigned field that already has known identifiability.

Conditions may be strong, but they may not be vacuous. “The only zero of `V`
is `p`”, “the relevant operator is injective”, or “the coefficient map has
trivial kernel” is not progress unless injectivity/triviality is reduced to
concrete, independently verifiable analytic or algebraic assumptions.

## Exact versus asymptotic claims

Never slide between these statements:

- pointwise `Vₚ,q(x) = 0` for every `x`;
- `Vₚ,q = 0` almost everywhere under `q`;
- an integrated loss such as `E_q ‖Vₚ,q(x)‖² = 0`;
- approximate smallness `‖V‖ ≤ ε`;
- sequential convergence `Vₚ,qₙ → 0`;
- weak, total-variation, Wasserstein, MMD, or coefficient convergence of
  distributions.

Name the drift norm and distribution topology explicitly. Use
`IdentifiesAtZero` for exact results and `AsymptoticallyIdentifies` for limit
results.

## Required research workflow

### 1. State the candidate

- Add a `CandidateSpec` in `CandidateConditions.lean`.
- State every quantifier and side condition.
- Give a short mechanism explaining why zero drift should determine `p` and
  `q`.
- Prove or exhibit `candidate.IsLegitimate`, including a distinct pair that
  satisfies the condition before zero drift is imposed.

### 2. Try to destroy it

Before attempting a proof, search for counterexamples in at least these
regimes when applicable:

- one- and two-point distributions;
- symmetric distributions and symmetric kernels;
- collapsed or disjoint supports;
- constant, separable, low-rank, and flat kernels;
- duplicated or insufficient probes;
- zero normalizers and boundary/support failures;
- finite-basis coefficient vectors with small `m`;
- dimension-count failures (`dN` too small for the interaction family);
- sequences with vanishing drift but escaping mass or oscillating measures.

Use `IsExactCounterexample`, `RefutesCondition`, and `PassesFamilyTest` to keep
the distinction between finite testing and a theorem explicit.

### 3. Log every failed route

Append to `LoggedFailures.md`:

- the exact condition;
- the intended proof mechanism;
- the smallest counterexample or precise proof obstruction;
- whether the failure is fatal or suggests a repair;
- the Lean definitions/files involved.

Do not silently delete rejected ideas. Check this log before proposing a new
condition to avoid cycling through equivalent failures.

### 4. Write the proof first

Only after stress testing, complete `WrittenProof.md` with:

- a formal theorem statement;
- all assumptions and where each is used;
- a lemma-by-lemma proof;
- treatment of normalization, null sets, support, and limiting operations;
- a dependency list distinguishing paper axioms, Mathlib results, and new
  lemmas;
- an explicit check that no step assumes uniqueness/identifiability.

If the written proof has a gap, return to stress testing. Do not use Lean to
paper over it.

### 5. Formalize in Lean

- Reuse `TrustedBoundary.lean` and `CounterexampleHarness.lean`.
- Prefer small named lemmas over a monolithic proof.
- Keep analytic side conditions visible rather than burying them in typeclass
  search.
- Use local `set_option` increases only when justified; never disable
  heartbeats globally to hide nontermination.
- Compile after each lemma and run `scripts/Check.ps1` before reporting success.

### 6. Audit the final theorem

Before claiming completion:

1. run `scripts/PrintAxioms.ps1 -Declaration Fully.Qualified.TheoremName`
   (equivalently, `#print axioms Fully.Qualified.TheoremName`);
2. verify every reported project axiom belongs to the reviewed allowlist in
   `Paperaxioms.lean`;
3. run the trust audit and full build;
4. re-read the theorem statement to ensure it proves the intended exact or
   asymptotic claim, not a weakened surrogate;
5. verify the candidate condition still admits a distinct pair independently
   of zero drift.

## Current verified route

The promoted result is the paper-native finite population theorem
`finitePopulationMeanShift_identifies` in `PopulationIdentifiability.lean`.
Its basis elements are measurable, nonnegative, unit-mass densities; their
mixtures are genuine probability measures. The proof connects the normalized
population mean-shift field to the density interaction numerator using nonzero
normalizers and equation (11), then applies the axiom-free minor algebra.

Practical versions must supply a positive `InteractionFrameBound`, not merely
assert unnamed injectivity. This bound implies nondegeneracy and yields the
explicit stability estimate `‖a-b‖₁ ≤ (2B/c)‖V_probes‖`. A necessary probe
dimension bound is also formalized. `finiteBasisCandidate` records the model
family and proves legitimacy from an independently unequal pair.

The result is deliberately scoped to the ideal population field, data-space
matching, and no CFG. Zero population energy reaches the pointwise hypothesis
only with integrability, continuity, and full topological support. A finite
minibatch estimator and a signed CFG target require separate theorems.

## Conditional modules

`CharacteristicIdentifiability.lean` and `GaussianNondegeneracy.lean` are
opt-in through `DriftingIdentifiability.Conditional`. They may guide research,
but cannot support a promoted success claim:

- characteristic-gradient embedding injectivity is substantively equivalent
  to the desired implication for the MMD drift;
- the Gaussian anti-symmetric extension is synthetic and is not proved equal
  to the paper's integral-induced interaction vectors;
- all Gaussian/RKHS declarations remain explicitly classified external axioms.

`scripts/AxiomAudit.ps1` rejects any promoted theorem that acquires one of
these conditional dependencies.

## Definition of success

Success requires all of the following:

- a condition that is concrete, nonvacuous, and independently checkable;
- no surviving counterexample in the declared stress-test scope;
- a complete written proof;
- a Lean theorem with no `sorry` and no new unreviewed axioms;
- a clean `#print axioms` audit and passing `scripts/Check.ps1`;
- a statement that precisely matches the claimed exact or asymptotic result.
