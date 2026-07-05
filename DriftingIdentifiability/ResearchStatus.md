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
- [ ] Discharge the nondegeneracy and integrability hypotheses for the paper's
      concrete kernel/probe system rather than assuming them.
- [ ] Bridge the density-valued statement to a measure-level `IdentifiesAtZero`
      claim, and probe-wise zero drift to global zero drift.
- [~] Quantitative coefficient stability proved (`FiniteStability.lean`): `ℓ¹`
      coefficient distance ≤ total minor mass, hence `minor mass → 0 ⟹
      coefficients → 0`. The remaining gap is bounding the minor mass by the
      drift norm (the linear-independence lower bound).
- [ ] Formulate and prove a genuinely asymptotic result with named norms and
      measure topology (`AsymptoticallyIdentifies`).

## Active candidate

Finite-basis interaction separation — complete and audited through the
drift-level statement. No open candidate at the finite/exact level; the next
work items are the four unchecked analytic/asymptotic obligations above.
