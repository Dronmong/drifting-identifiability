# Written proof: finite population mean-shift identifiability

## Theorem and scope

Let `μ` be a reference probability measure and let `φ₁,…,φₘ` be measurable,
nonnegative, integrable densities satisfying `∫φᵢ dμ = 1`. For simplex
coefficients `a,b`, define the genuine probability measures

```text
p = (∑ᵢ aᵢφᵢ) · μ,     q = (∑ᵢ bᵢφᵢ) · μ.
```

Fix a mean-shift kernel and finitely many probes. Assume all integrals in
equations (11) and (31) are valid, both normalizers are nonzero at every probe,
and the strict-pair interaction vectors have a positive frame lower bound.
Then pointwise zero of the normalized population mean-shift field implies
`p=q`.

This statement is about the ideal population field in data space. It excludes
CFG/signed targets and does not identify the finite-batch estimator with the
population field.

## Proof

1. Each mixture density is measurable, nonnegative, integrable, and has
   integral one. Therefore `Measure.withDensity` produces probability measures
   `p` and `q`.
2. Applying the standard `withDensity` integration identity twice proves that
   the measure-level interaction integral against `p.prod q` equals the
   density-weighted iterated integral against `μ`.
3. Equation (11) writes normalized mean-shift drift as the inverse product of
   its two nonzero normalizers times the interaction numerator. Hence zero
   normalized drift forces the numerator to vanish at every probe.
4. Equation (31) expands the stacked numerator as
   `∑ᵢ∑ⱼ aᵢbⱼ Uᵢⱼ`.
5. Anti-symmetry groups this ordered sum into
   `∑_{i<j}(aᵢbⱼ-aⱼbᵢ)Uᵢⱼ`. This grouping is proved in Lean rather than
   axiomatized.
6. A positive frame bound implies linear independence, so every minor
   `aᵢbⱼ-aⱼbᵢ` is zero.
7. Probability normalization gives
   `aᵢ-bᵢ = ∑ⱼ(aᵢbⱼ-aⱼbᵢ)=0`; therefore `a=b`.
8. Equal coefficients give definitionally equal mixture measures, hence
   `p=q`.

No step assumes uniqueness of a zero-drift equilibrium or injectivity of the
drift-to-distribution map.

## Population-loss bridge

For the nonnegative function `x ↦ ‖Vₚ,q(x)‖²`, an integrable zero population
integral implies `Vₚ,q=0` almost everywhere under `q`. If the field is
continuous and `q` assigns positive measure to every nonempty open set,
continuous functions equal almost everywhere are equal everywhere. The main
theorem then applies.

Without full support this upgrade is false; `FailureCases.lean` gives the
continuous identity field under `dirac 0` as a counterexample.

## Stability

Let `c>0` satisfy

```text
c ∑_{i<j}|zᵢⱼ| ≤ ‖∑_{i<j} zᵢⱼ Uᵢⱼ‖.
```

The ordered minor mass is twice the strict-pair mass, while normalized
coefficients satisfy

```text
‖a-b‖₁ ≤ ∑ᵢ∑ⱼ |aᵢbⱼ-aⱼbᵢ|.
```

If the absolute normalizer product is at most `B` at every probe, then

```text
‖a-b‖₁ ≤ (2B/c) ‖normalized probe drift‖.
```

The same uniform estimate proves coefficient convergence from probe-drift
convergence. Nondegeneracy also requires
`card(StrictPair m) ≤ N · dim(E)`, which is now machine checked.

## Concrete general finite family

Let `m≥2`, let `z₀,…,zₘ₋₁∈ℝ` be distinct, and let the reference law be the
uniform empirical distribution on these points. Define the `i`th basis density
to equal `m` at `zᵢ` and zero at every other support point. These functions are
measurable, nonnegative, and have unit integral, so their simplex mixtures are
genuine probability measures.

For the unit-bandwidth Gaussian kernel and the mean-shift interaction, direct
finite integration gives

```text
Uᵢⱼ(n) = k(n,zᵢ)k(n,zⱼ)(zᵢ-zⱼ),
```

using one integer probe `n` for each strict pair. Expanding the Gaussian gives

```text
Uᵢⱼ(n)
 = exp(-n²)
   [exp(-(zᵢ²+zⱼ²)/2)(zᵢ-zⱼ)]
   [exp(zᵢ+zⱼ)]ⁿ.
```

The first factor is nonzero for every probe. The second is nonzero because the
support points are distinct. If all strict-pair sums `zᵢ+zⱼ` are distinct, the
geometric bases `exp(zᵢ+zⱼ)` are distinct. The square evaluation matrix is a
Vandermonde matrix and is nonsingular, so the actual induced vectors are
linearly independent. Finite-dimensional norm equivalence then supplies a
positive frame constant.

This proves `gaussianEmpiricalPoint_identifies`: no frame or injectivity
conclusion is assumed by its caller. Its remaining practical limitation is
that the selected frame constant is existential rather than a useful explicit
lower bound. Since the unit Gaussian satisfies `0<k≤1`, each normalizer lies in
`(0,1]`, so the product bound is explicitly `B=1`; the concrete stability
estimate is therefore `‖a-b‖₁ ≤ (2/c)‖V_probes‖`.

## Legitimacy and failure audit

The registered `finiteBasisCandidate` admits distinct laws whenever two
basis-induced mixtures are unequal; the first two simplex vertices give a
convenient witness for `m≥2`. This is established before imposing zero drift.

Formal regressions cover zero normalizers, collapsed bases, duplicated
interaction vectors, insufficient probe dimension, missing full support, and
non-injective feature maps. A measurable embedding is sufficient to lift
feature-law equality back to source-law equality.

## Lean map and dependencies

- Probability measures and normalized bridge:
  `PopulationIdentifiability.lean`.
- Algebraic grouping and exact coefficient proof: `FiniteGrouping.lean` and
  `PaperFiniteIdentifiability.lean`.
- Quantitative frame estimates: `FiniteStability.lean`.
- Concrete empirical bases and the axiom-free Gaussian/Vandermonde theorem:
  `EmpiricalFrameBound.lean`.
- Candidate legitimacy and regressions: `PopulationCandidate.lean` and
  `FailureCases.lean`.

The promoted theorem uses only the reviewed paper facts
`equation_11_bilinear_mean_shift`,
`equation_31_bilinear_expansion`, and
`antisymmetric_kernel_induces_basis_antisymmetry`, besides foundational
Mathlib axioms. It has no dependency on characteristic-kernel, Gaussian
metrization, or Gaussian Gram axioms; this is enforced by `AxiomAudit.ps1`.
