# Research status

## Infrastructure

- [x] Paper definitions and results formalized behind a reviewed boundary.
- [x] Exact and asymptotic theorem targets distinguished.
- [x] Candidate nonvacuity/anti-circularity predicates available.
- [x] Counterexample and finite-coefficient harness available.
- [x] Trust audit rejects new axioms and incomplete proofs.
- [x] Warning-free whole-library build configured.

## Mathematical baseline

- [x] Anti-symmetry proves `p = q → V = 0`.
- [x] Appendix C.1 finite expansion and zero-minor conclusion exposed.
- [x] Coefficient-minor vanishing **proved from scratch** (`FiniteGrouping.lean`,
      `coefficientMinorsVanish_of_antisymm`): the finite exact route
      (`finiteBasisDensitiesEqual`) is now axiom-free, no longer using the paper
      axiom `zero_drift_coefficient_minors`.
- [x] Normalized probability coefficient vectors with all minors zero are equal
      (`normalizedParallelCoefficientsAreEqual`).
- [x] Coefficient equality lifts to density equality using basis independence
      (`finiteBasisDensitiesEqual`); the distinctness direction is also proved
      (`basisDensity_injective_of_basisIndependent`).
- [x] Finite condition is legitimate: a distinct pair exists before zero drift
      and the separation premises are satisfiable (`FiniteLegitimacy.lean`).
- [x] Actual density-interaction drift-zero (probe-wise) identifies the basis
      densities via the reviewed paper axioms
      (`driftProbeZeroIdentifiesDensities`,
      `meanShiftInteractionIdentifiesDensities`).
- [x] Nondegeneracy discharged for a concrete Gaussian-kernel interaction system
      (`GaussianNondegeneracy.lean`, `gaussianInteractionIdentifiesCoefficients`):
      reduced to Micchelli's strict positive definiteness of the Gaussian
      (`gaussian_gram_linearIndependent`) via an axiom-free anti-symmetric
      extension, so the finite identifiability holds with nondegeneracy derived
      rather than assumed. Remaining: match the paper's exact integral-induced
      interaction vectors (analytic), and the integrability side conditions.
- [x] Measure-level `IdentifiesAtZero` reached for a concrete paper field
      (`CharacteristicIdentifiability.lean`): for a characteristic kernel — the
      Gaussian being the witnessed instance — zero MMD drift (equation 41)
      between probability measures forces `p = q`. Identifiability is reduced to
      the reviewed RKHS embedding-injectivity axioms; the drift→embedding step is
      definitional for the MMD field (see the transparency note in
      `Paperaxioms.lean`).
- [ ] Bridge the *finite-basis* density statement to the measure level, and
      probe-wise zero drift to global zero drift.
- [~] Quantitative coefficient stability proved (`FiniteStability.lean`): `ℓ¹`
      coefficient distance ≤ total minor mass, hence `minor mass → 0 ⟹
      coefficients → 0`. The remaining gap is bounding the minor mass by the
      drift norm (the linear-independence lower bound). Concrete plan:
      (1) prove the general anti-symmetric grouping identity
      `∑ᵢ∑ⱼ aᵢbⱼ • Uᵢⱼ = ∑_{i<j} (aᵢbⱼ - aⱼbᵢ) • Uᵢⱼ` (axiom-free algebra;
      currently only the zero case is exposed as `equation_32_grouped_zero_drift`);
      (2) package `c ↦ ∑ c_p • U_p` as a `LinearMap` whose kernel is `⊥` from
      nondegeneracy, and apply `LinearMap.exists_antilipschitzWith` (finite
      dimension) to get `‖minor‖ ≤ K ‖drift‖`. Chaining with the stability bound
      yields Lipschitz stability `ℓ¹ coeff distance ≤ C ‖drift‖`.
- [x] Asymptotic result with named discrepancy and topology
      (`gaussianMmd_asymptoticallyIdentifies`): an `AsymptoticallyIdentifies`
      instance for the Gaussian, using the MMD discrepancy (equation 37) as the
      drift size and the Lévy–Prokhorov metric as the distribution distance, via
      the reviewed metrization axiom `gaussian_mmd_metrizes_weakConvergence`.
      Caveat: the drift size is the MMD discrepancy, not the raw drift-field
      norm — that connection is subtle and is logged, not claimed.

## Active candidate

Finite-basis interaction separation — complete and audited through the
drift-level statement. No open candidate at the finite/exact level; the next
work items are the four unchecked analytic/asymptotic obligations above.
