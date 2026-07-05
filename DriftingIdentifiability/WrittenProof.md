# Written proof workspace

## Candidate and status

- **Candidate name:** finite-basis interaction separation.
- **Status:** stress-tested; finite algebraic proof complete; formalized in
  Lean including the legitimacy witnesses (`FiniteLegitimacy.lean`).
- **Claim:** exact finite-model identifiability. No asymptotic claim is made
  here.

## Formal theorem statement

Let `a,b ∈ ℝᵐ` be nonnegative coefficient vectors with

```text
∑ᵢ aᵢ = ∑ᵢ bᵢ = 1.
```

Let `Uᵢⱼ ∈ Eᴺ` satisfy `Uᵢⱼ = -Uⱼᵢ`, and suppose the family
`{Uᵢⱼ : i<j}` is linearly independent over `ℝ`. If

```text
∑ᵢ ∑ⱼ aᵢ bⱼ Uᵢⱼ = 0,
```

then `a=b`. Consequently, for any common finite basis `φ₁,…,φₘ`, the densities
`p=∑aᵢφᵢ` and `q=∑bᵢφᵢ` are equal.

For the paper’s drifting field, equation (31), anti-symmetry, and the stated
linear-independence condition supply the displayed bilinear hypotheses.

## Legitimacy witness

The condition does not imply `p=q` before zero drift is imposed. For `m≥2`,
the distinct simplex vertices `a=e₀` and `b=e₁` are both normalized,
nonnegative coefficient vectors. The model-class and interaction-separation
conditions constrain the basis/kernel/probes, not the equality of these two
coefficient vectors.

This is now machine-checked in `FiniteLegitimacy.lean`, not merely argued:

- `stdBasisProbVector m k` builds the simplex vertex `eₖ` as a
  `FiniteProbabilityVector`; `exists_distinct_probVectors (hm : 2 ≤ m)`
  produces the distinct pair `e₀ ≠ e₁`.
- `basisDensity_injective_of_basisIndependent` shows the distinct coefficients
  lift to distinct densities under `DensityBasisIndependent φ`, so the witness
  is not an artifact of the coefficient encoding.
  `finiteConditionAllowsDistinctDensities` packages the two into a
  density-level `ConditionAllowsDistinctPair`.
- `exists_separation_with_distinct_pair` exhibits a concrete interaction system
  `witnessU i j = i - j` on `m=2, N=1` that is *simultaneously* anti-symmetric
  (`witnessU_anti`) and nondegenerate (`witnessU_nondegenerate`), yet still
  admits `a ≠ b`. Only adding `zeroBilinear` forces equality, so the separation
  premises encode no hidden `a=b`.

`#print axioms` on `finiteConditionAllowsDistinctDensities` and
`exists_separation_with_distinct_pair` reports only
`propext, Classical.choice, Quot.sound` — in particular **not**
`zero_drift_coefficient_minors`. The legitimacy results are therefore
provably independent of the paper's zero-drift axiom, as legitimacy requires.

The witness is exact only for `2 ≤ m`; the `m ≤ 1` collapse is logged in
`LoggedFailures.md`.

## Stress tests completed

- **One/two point:** `m=1` is necessarily trivial; for `m=2`, a single nonzero
  `U₀₁` suffices and the coefficient calculation is exact.
- **Missing normalization:** fails for proportional vectors, e.g.
  `(1,0)` and `(2,0)`; logged in `LoggedFailures.md`.
- **Degenerate interaction/probes:** fails when every `Uᵢⱼ=0`; logged.
- **Flat kernel:** reduces to mean matching and fails even for distinct smooth
  full-support laws with the same mean; logged.
- **Collapsed/disjoint supports:** do not affect the algebra once the finite
  representation and observable-vector independence are assumed; validating
  those analytic assumptions remains a separate obligation.
- **Dimension count:** full independence requires enough ambient probe
  dimensions to host `m(m-1)/2` vectors. The paper’s heuristic `dN ≫ m²` is a
  plausible way to make this possible, not by itself a proof of independence.
- **Asymptotic sequences:** not claimed in this theorem. A quantitative lower
  singular-value bound will be needed for stability estimates.

## Definitions and assumptions

1. **Anti-symmetry of `U`:** groups the ordered bilinear sum into minors.
2. **Linear independence of `{Uᵢⱼ : i<j}`:** forces every minor
   `aᵢbⱼ-aⱼbᵢ` to vanish.
3. **Equal unit mass:** removes the scale ambiguity left by vanishing minors.
4. **Nonnegativity:** is needed for the probability interpretation, but the
   final algebraic equality uses only equal nonzero total mass.
5. **Finite basis representation:** turns coefficient equality into density
   equality. Basis independence is needed to make coefficients an identifiable
   parametrization, though the forward implication `a=b → p=q` is immediate.

## Lemmas

### Lemma 1 — zero drift gives zero minors

Under anti-symmetry and linear independence of the interaction vectors, the
bilinear zero equation implies

```text
aᵢbⱼ-aⱼbᵢ = 0   for all i,j.
```

This is exactly the reviewed paper result
`Paper.zero_drift_coefficient_minors`.

### Lemma 2 — normalized vectors with zero minors are equal

Fix `i` and sum the minor identity over `j`:

```text
0 = ∑ⱼ(aᵢbⱼ-aⱼbᵢ)
  = aᵢ(∑ⱼbⱼ) - (∑ⱼaⱼ)bᵢ
  = aᵢ-bᵢ.
```

Thus `aᵢ=bᵢ` for every `i`, hence `a=b`. This proof avoids selecting a positive
coordinate and remains valid for every finite `m`; when `m=0`, normalized
coefficient vectors have no inhabitants.

### Lemma 3 — equal coefficients give equal basis densities

Substitute `a=b` into `p(y)=∑aᵢφᵢ(y)` and
`q(y)=∑bᵢφᵢ(y)`.

## Main proof

Apply Lemma 1 to the zero bilinear drift. Apply Lemma 2 using probability
normalization to obtain `a=b`. Lemma 3 yields equality of the represented
density functions. No uniqueness or identifiability statement is assumed.

## Anti-circularity audit

- No step assumes `V=0 → p=q` or its contrapositive.
- Interaction-vector independence is a concrete finite linear-algebra
  condition, not equality of distributions. Its validation for a particular
  kernel/probe system remains explicit work.
- The model assumptions admit distinct coefficient vectors independently of
  zero drift.
- Exact equality and convergence are not conflated.
- Analytic issues (Fubini, normalizers, support, and passage from sampled to
  pointwise zero drift) are supplied by the paper interface or left as named
  future obligations; they are not silently assumed in this algebraic result.

## Lean implementation map

| Written lemma | Lean declaration | File | Status |
|---|---|---|---|
| Zero drift gives zero minors | `coefficientMinorsVanish` | `PaperFiniteIdentifiability.lean` | proved |
| Normalized zero-minor vectors agree | `normalizedParallelCoefficientsAreEqual` | `PaperFiniteIdentifiability.lean` | proved |
| Finite coefficient identifiability | `finiteCoefficientIdentifiable` | `PaperFiniteIdentifiability.lean` | proved |
| Equal coefficients give equal basis densities | `finiteBasisDensitiesEqual` | `PaperFiniteIdentifiability.lean` | proved |
| Simplex-vertex probability vectors | `stdBasisProbVector` | `FiniteLegitimacy.lean` | proved |
| Distinct pair exists for `2 ≤ m` | `exists_distinct_probVectors` | `FiniteLegitimacy.lean` | proved |
| Distinct coefficients give distinct densities | `basisDensity_injective_of_basisIndependent` | `FiniteLegitimacy.lean` | proved |
| Density-level legitimacy (distinct pair) | `finiteConditionAllowsDistinctDensities` | `FiniteLegitimacy.lean` | proved |
| Separation hypothesis is satisfiable | `exists_separation_with_distinct_pair` | `FiniteLegitimacy.lean` | proved |
| Actual drift-zero ⇒ grouped bilinear zero | `DriftFiniteSetup.zeroBilinear` | `PaperDriftIdentifiability.lean` | proved |
| Probe-wise drift-zero identifies densities | `driftProbeZeroIdentifiesDensities` | `PaperDriftIdentifiability.lean` | proved |
| Mean-shift kernel specialization | `meanShiftInteractionIdentifiesDensities` | `PaperDriftIdentifiability.lean` | proved |

## Drift-level connection (Appendix C.1, equations 30–32)

The table's first four rows start from the *grouped* hypothesis
`∑ᵢ∑ⱼ aᵢbⱼ • Uᵢⱼ = 0`. `PaperDriftIdentifiability.lean` closes the gap to the
*actual* field: `DriftFiniteSetup` bundles an anti-symmetric interaction kernel
`K`, basis `φ`, probes, probability vectors, nondegeneracy, and the hypothesis
that the density-interaction drift (equations 28/30) vanishes at every probe.
`equation_31_bilinear_expansion` rewrites that drift as the grouped bilinear
sum, and `antisymmetric_kernel_induces_basis_antisymmetry` supplies the
anti-symmetry of the induced `U`; together they reduce the drift-level setup to
`FiniteCoefficientSetup`, after which the finite algebra applies unchanged.

`#print axioms driftProbeZeroIdentifiesDensities` reports only the standard
Mathlib axioms plus the reviewed paper axioms `equation_31_bilinear_expansion`,
`antisymmetric_kernel_induces_basis_antisymmetry`, and
`zero_drift_coefficient_minors` — all on the `Paperaxioms.lean` allowlist. The
zero-drift hypothesis is probe-wise (finite `N` probes), matching Appendix C.1;
passing to global zero drift and to a measure-level statement remain open.

## Expected `#print axioms` output

The final finite theorem may depend on Mathlib’s standard foundational axioms
and the reviewed paper axiom `Paper.zero_drift_coefficient_minors`. It must not
depend on any new project axiom.
