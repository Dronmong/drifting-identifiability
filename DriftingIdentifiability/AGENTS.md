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
dimension bound is also formalized. `finiteBasisCandidate` records the
preliminary model-family condition. The accepted
`finitePopulationMeanShiftCandidate` additionally packages the full
`PopulationMeanShiftFiniteSetup`; its condition includes the fixed kernel,
probes, regularity, integrability, and positive frame certificate, and
`finitePopulationMeanShiftCandidate_identifiesAtZero` proves the canonical
`IdentifiesAtZero` target. Its legitimacy theorem still requires an
independently unequal represented pair before zero drift is assumed.

`EmpiricalFrameBound.lean` now supplies an axiom-free concrete general-`m`
instance on `ℝ`: a uniform empirical point-density basis, unit Gaussian kernel,
integer probes, distinct support points, and distinct strict-pair sums. A
Vandermonde proof establishes nondegeneracy for the actual integral-induced
vectors and `gaussianEmpiricalPoint_identifies` packages the full population
theorem. The main concrete setup uses the explicit inverse-interaction-matrix
certificate `gaussianEmpiricalPointCertifiedFrameConstant`. It is a proved
positive lower frame constant, but positivity alone must not be described as
good numerical conditioning; report or bound its actual value for such claims.

The deterministic finite probe loss is `probeDriftEnergy`. Zero loss is
equivalent to zero at every selected probe, and coefficient error is controlled
by its square root. Do not identify this ideal finite population quantity with
Algorithm 2's random minibatch estimator without a separate finite-sample
theorem.

`PracticalModelClasses.lean` contains the Objective 3 layer. The structured
Gaussian theorem now works in separable real inner-product spaces at arbitrary
positive bandwidth. Arbitrary/adaptive probes are accepted only with an
explicit `InteractionDualCertificate`; never paraphrase that finite
biorthogonality certificate as an automatically available learned feature.
Continuous and smooth bases enter through a `δ<c` interaction-perturbation
bound in the generic route. `SmoothBumpBasis.lean` is the independently proved
exception: it directly certifies a non-atomic two-component `C∞` basis by an
ordered-support sign argument. Do not generalize that result to arbitrary
smooth bases or more than two components. Its promoted Gaussian setup requires
`ValidBandwidth`. The concrete Laplace result is currently a two-atom,
one-probe theorem, not a general Laplace identifiability theorem.

The result is deliberately scoped to the ideal population field, data-space
matching, and no CFG. Zero population energy reaches the pointwise hypothesis
only with integrability, continuity, and full topological support. A finite
minibatch estimator and a signed CFG target require separate theorems.

Objective 4 now has promoted **fixed-anchor/sample-split** finite-sample
infrastructure:
`FiniteSampleBridge.lean` propagates
estimator MSE into coefficient error, `Algorithm2Estimator.lean` proves the
safe algebraic/boundedness facts for Algorithm 2's `compute_V` (including the
mass-product times self-normalized-centroid-difference form), and
`SelfNormalizedConsistency.lean` proves a generic self-normalized centroid MSE
bound from the reviewed sample-mean theorem. `Algorithm2SNIS.lean` now
instantiates that ratio theorem for the fixed-anchor, `selfMask=false`
Algorithm-2 centroids: the row-softmax factor cancels and the centroids are
SNIS estimators with the column-reweighted weight
`sqrt(k(x_i,y) * k(x_i,y) / sum_r k(x_r,y))`. `ColumnReweightedMeanShift.lean`
then treats that weight as the population mean-shift kernel and proves
identifiability/stability under an explicit modified-kernel
`InteractionFrameBound`; `interactionFrameBound_of_strictPairScaling` is the
preferred transfer tool when the modified vectors are positive pairwise
rescalings of a certified baseline family. `ColumnReweightedTwoAtom.lean`
discharges that frame condition concretely for the two-atom empirical basis:
`algorithm2Kernel` is definitionally the paper Laplace kernel, the atoms pin
the reweighting factor so the modified interaction vector is an exact
strict-pair rescaling `U^col = (1/sqrt(g(0)g(1))) • U^bare` of the bare one,
and `columnReweighted01Setup` certifies the sharp frame constant directly from
kernel positivity (with `B = 1` at the anchors and an explicit `≥ bare/N`
conditioning floor). Do not generalize that certificate beyond the two-atom
class without a new frame proof. `SelfMaskPerturbation.lean` now formalizes the
implementation mask as an explicit deterministic perturbation of the
leave-masked-out/deleted estimator: masked logits equal the no-mask logits
times `exp(-1000000/temperature)` on masked entries, every masked affinity is
within the proved `maskAffinityErrorBound` of its deleted counterpart, and the
drifts differ by at most `4*Npos*Nneg*R0*eta`. This deliberately does not compare
the masked estimator to the full no-mask estimator on the same reused samples.
`DeletedEstimatorConsistency.lean` supplies the statistical theorem for the
deleted estimator: `deletedDrift` factors as a mass product times a centroid
difference; the deleted positive centroid provably equals the no-mask positive
centroid (positives are never masked), so its SNIS bound transports; the
deleted negative centroid satisfies the indexed bias-tolerant ratio theorem
`selfNormalizedIndexed_meanSquare_le` with per-slot weight functions
(`deletedNegativeColumnWeight`), per-slot leave-out mean shifts bounded by `b`,
and an abstract denominator floor `dmin`. `DenominatorTail.lean` separately
provides the high-probability replacement for the deterministic denominator
floor: `selfNormalizedIndexed_deviation_prob_le` uses checkable denominator
means/variances and an explicit split `0 < t < Σ μw`. Do not silently set
`b = 0` for the eye mask (the per-slot leave-out targets genuinely differ),
do not assume `Σ μw` is large without data, and do not present the
mean-square or denominator-tail bounds as identifiability claims: they feed the
estimator-agnostic bridge through estimator-error control only.

These probability theorems do not cover the paper's exact reuse
`anchors(ω) = Yneg(ω)`. Their theorem signatures take deterministic anchors
separately from the random negative batch. Substitution would make the weight
functions random and jointly batch-dependent, invalidating the fixed-weight
SNIS hypotheses. Do not call Objective 4 complete for the paper estimator until
that coupled random-anchor concentration theorem is proved.

Objective 5 is now initialized in `FeatureSpaceIdentifiability.lean`. The safe
default conclusion of any feature-space theorem is equality of feature laws
`φ♯p = φ♯q`; source-law equality is promoted only through an explicit
`MeasurableEmbedding φ` hypothesis (or one embedded feature in a finite feature
family). A non-injective feature collision is formalized by distinct source
Dirac laws with equal feature laws. Heterogeneous feature families are supported
by `HeterogeneousFeatureFamily`; exact lifting may use an independently proved
`MeasureDetermining` condition, and approximate lifting must pass through a
`FeatureStabilityCertificate` controlling a stated source discrepancy by stated
feature discrepancies. Do not turn feature-space matching into `p=q` unless one
of these lifting hypotheses is present and independently justified.

Objective 6 is initialized in `CFGAffine.lean`. CFG is treated as an affine
density/coefficient problem, not as an automatic probability-measure theorem.
Use `FiniteAffineVector` for coefficients that sum to one but may be negative.
`CFGDriftFiniteSetup.generated_eq_cfgTarget` proves the safe CFG conclusion:
zero drift matching the effective negative density `(1-γ)q + γu` to the
conditional density `p` implies `q = (1/(1-γ))p - ((1/(1-γ))-1)u`. Do not
paraphrase this as `q=p`, and do not convert the affine target to a probability
law unless `CFGTargetNonnegative` (or an equivalent concrete nonnegativity
proof) has been supplied.

Objective 7 numerical diagnostics live in `numerics/`. `run_all.py` is the
synthetic/paper-parameter ledger; `real_feature_diagnostics.py` consumes
externally supplied `.npy`/`.npz` encoder features and reports the actual
finite interaction-matrix conditioning, dual-certificate constants, softmax
ESS, and column-reweighted diagnostics. Do not present synthetic diagnostics
as evidence about the paper's real encoder, and do not claim the real-feature
evaluation has been run unless an actual feature tensor or checkpoint is
present.

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
